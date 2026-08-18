import os
import json
import time
import torch
import queue
import logging
import soundfile
import threading
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import bacpipe
from bacpipe.core.audio_processor import AudioHandler
from bacpipe.core.experiment_manager import (
    replace_default_kwargs_with_user_kwargs,
)

logger = logging.getLogger("bacpipe")


class Embedder(AudioHandler):
    """
    This class takes care of loading the specified model and using it
    to process the audio data to create embeddings.
    This class is also used to create dimensinoality reductions from
    embeddings.
    At the end if instantiation, the selected model is loaded and the
    model is associated with the specified device.
    kwargs that are not specifically passed will be taken from
    bacpipe.config and bacpipe.settings.

    Parameters
    ----------
    AudioHandler : class
        Helper class that handles loading of audio
    """

    def __init__(
        self,
        model_name,
        loader=None,
        CustomModel=None,
        dim_reduction_model=False,
        audio_dir=None,
        **kwargs,
    ):
        """
        This class takes care of loading the specified model and using it
        to process the audio data to create embeddings.
        This class is also used to create dimensinoality reductions from
        embeddings.
        At the end if instantiation, the selected model is loaded and the
        model is associated with the specified device.
        kwargs that are not specifically passed will be taken from
        bacpipe.config and bacpipe.settings.

        Parameters
        ----------
        model_name : str
            name of selected embedding model
        loader : Loader object
            Object that has all the necessary path information and methods
            to load and save all the processed data
        CustomModel : class, optional
            custom model class to use for processing, by default None
        dim_reduction_model : bool, optional
            Can be bool or the string corresponding to the
            dimensionality reduction model, by default False
        audio_dir : str, optional
            path to the directory containing the audio files, by default None
        """
        self.file_length = {}
        self.loader = loader
        if loader:
            audio_dir = loader.audio_dir
        elif not audio_dir is None:
            audio_dir = audio_dir

        self.dim_reduction_model = dim_reduction_model
        if dim_reduction_model:
            self.dim_reduction_model = True
            self.model_name = dim_reduction_model
        else:
            self.model_name = model_name

        kwargs = replace_default_kwargs_with_user_kwargs(
            ["audio_dir", "models", "dim_reduction_model"], **kwargs
        )
        self.nr_parallel_workers = kwargs.get("nr_parallel_workers")

        self._init_model(
            dim_reduction_model=dim_reduction_model,
            CustomModel=CustomModel,
            **kwargs,
        )
        super().__init__(
            model=self.model,
            audio_dir=audio_dir,
            **kwargs,
        )
        if self.model.bool_classifier:
            self.classifier = Classifier(
                self.model,
                model_name,
                audio_dir=audio_dir,
                use_folder_structure=(
                    loader.use_folder_structure if loader else False
                ),
                **kwargs,
            )

    def _init_model(self, CustomModel=None, **kwargs):
        """
        Load the model specific module, instantiate the model and allocate
        the device for the model.

        Parameters
        ----------
        CustomModel : class, optional
            custom model class to use for processing, by default None
        """
        if self.dim_reduction_model or CustomModel is None:
            if self.dim_reduction_model:
                module = importlib.import_module(
                    f"bacpipe.model_pipelines.dimensionality_reduction.{self.model_name}"
                )
            else:
                module = importlib.import_module(
                    f"bacpipe.model_pipelines.feature_extractors.{self.model_name}"
                )
            self.model = module.Model(model_name=self.model_name, **kwargs)
        else:
            self.model = CustomModel(model_name=self.model_name, **kwargs)
        self.model.prepare_inference()

    def init_dataloader(self, audio):
        """
        Create a dataloader from the audio samples based on the model type.

        Parameters
        ----------
        audio : torch.Tensor or tf.Tensor
            preprocessed audio samples

        Returns
        -------
        torch.utils.data.DataLoader or tf.data.Dataset
            batched dataloader or dataset with batch size of the model
        """
        if "tensorflow" in str(type(audio)):
            import tensorflow as tf

            return tf.data.Dataset.from_tensor_slices(audio).batch(
                self.model.batch_size
            )

        elif "torch" in str(type(audio)):

            return torch.utils.data.DataLoader(
                audio, batch_size=self.model.batch_size, shuffle=False
            )

    def batch_inference(self, batched_samples, callback=None):
        """
        Run the model on all batches and collect the embeddings.

        Parameters
        ----------
        batched_samples : iterable
            iterable with the batched samples, created by init_dataloader
        callback : callable, optional
            callback function that is called with the fraction of processed
            batches, by default None

        Returns
        -------
        torch.Tensor or np.array
            embeddings for all batches
        """
        if self.model_name in bacpipe.TF_MODELS:
            import tensorflow

        embeds = []
        total_batches = len(batched_samples)

        for idx, batch in enumerate(
            tqdm(
                batched_samples,
                desc=" processing batches",
                position=0,
                leave=False,
            )
        ):
            with torch.no_grad():
                if self.model.device == "cuda" and hasattr(batch, "cuda"):
                    batch = batch.cuda()
                embeddings = self.model(batch)
                if self.model.bool_classifier:
                    try:
                        self.classifier.classify(embeddings)
                    except Exception as e:
                        logger.exception(
                            f"\nEmbeddings were created but classification failed due to error, {str(e)}"
                        )

            if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            embeds.append(embeddings)

            # callback with progress if progressbar should be updated
            if callback and total_batches > 0:
                fraction = (idx + 1) / total_batches
                callback(fraction)

        if self.model.bool_classifier:
            self.classifier.predictions = self.classifier.predictions.cpu()

        if isinstance(embeds[0], torch.Tensor):
            return torch.cat(embeds, axis=0)
        else:
            import tensorflow as tf

            return_embeds = tf.concat(embeds, axis=0).numpy()
            if len(return_embeds.shape) > 3:
                return_embeds = return_embeds.squeeze()
            return return_embeds

    def get_embeddings_for_audio(self, sample):
        """
        Create a dataloader for the processed audio frames and
        run batch inference. Both are methods of the self.model
        class, which can be found in the utils.py file.

        Parameters
        ----------
        sample : torch.Tensor
            preprocessed audio frames

        Returns
        -------
        np.array
            embeddings from model
        """
        batched_samples = self.init_dataloader(sample)
        embeds = self.batch_inference(batched_samples)
        if not isinstance(embeds, np.ndarray):
            try:
                embeds = embeds.numpy()
            except:
                try:
                    embeds = embeds.detach().numpy()
                except:
                    embeds = embeds.cpu().detach().numpy()
        return embeds

    def get_reduced_dimensionality_embeddings(self, embeds):
        """
        Apply the dimensionality reduction model to a set of embeddings.

        Parameters
        ----------
        embeds : np.array
            embeddings to be reduced

        Returns
        -------
        np.array
            reduced dimensionality embeddings
        """
        samples = self.model.preprocess(embeds)
        if "umap" in self.model.__module__:
            if samples.shape[0] <= self.model.umap_config["n_neighbors"]:
                logger.warning(
                    "Not enough embeddings were created to compute a dimensionality"
                    " reduction with the chosen settings. Please embed more audio or "
                    "reduce the n_neighbors in the umap config."
                )
        return self.model(samples)

    def run_dimensionality_reduction_pipeline(self):
        """
        Run the full dimensionality reduction pipeline for all embeddings
        and save the reduced embeddings to disk.

        This method checks the total number of embeddings and subsamples
        them if necessary to avoid running into memory errors.
        """
        if self.loader.metadata_dict["nr_embeds_total"] > 300_000:
            self.nr_subsampled_embeds_for_umap = 300_000
            logger.info(
                "Your dataset is very large, with a total of "
                f"{self.loader.metadata_dict['nr_embeds_total']}. "
                "Because umap requires loading all embeddings into memory to then "
                "calculate the low dimensional manifold, this is likely to cause "
                "memory errors. Instead bacpipe will subsample a random sample of "
                f"{self.nr_subsampled_embeds_for_umap} embeddings from your dataset. "
                "calculate a umap transformation based on those files and then apply "
                "the learned transformation to your entire dataset. It will not be "
                "super quick, but it will give you a 2d visualization for your "
                "dataset and it should prevent you running into out-of-memory problems."
            )
            use_sample_of_files = True
            sample_file_size = int(
                self.nr_subsampled_embeds_for_umap
                / int(
                    np.mean(
                        self.loader.metadata_dict["files"][
                            "nr_embeds_per_file"
                        ]
                    )
                )
            )
            sample_files = np.random.permutation(self.loader.files)[
                :sample_file_size
            ]
        else:
            use_sample_of_files = False
            sample_files = self.loader.files

        embeddings_list = []
        for file in tqdm(
            sample_files, desc="loading files", position=1, leave=False
        ):
            embeddings_list.append(self.loader.read_embedding_file(file))

        embeddings = np.concatenate(embeddings_list, axis=0)

        if use_sample_of_files:
            self.get_embeddings_from_model(embeddings)
            dim_reduced_embeddings_list = []
            for file in tqdm(
                self.loader.files,
                desc="processing files",
                position=1,
                leave=False,
            ):
                dim_reduced_embeddings_list.append(
                    self.model.model.transform(
                        self.loader.read_embedding_file(file)
                    )
                )

            dim_reduced_embeddings = np.concatenate(
                dim_reduced_embeddings_list, axis=0
            )
        else:
            try:
                dim_reduced_embeddings = self.get_embeddings_from_model(
                    embeddings
                )
            except NameError as e:
                error_string = (
                    "\n No embeddings found to process dimensionality reduction. It seems like "
                    "there was an error when initially calculating embeddings. See the error "
                    "logs in the `logs` directory or previous error messages in the terminal. \n"
                )
                logger.exception(f"error_string {str(e)}")
                raise NameError(error_string)

        self.loader.save_embedding_file(file, dim_reduced_embeddings)

    def embeddings_using_multithreading(self, array_of_audios):
        """
        Generate embeddings for all files in a pipelined manner:
        - Producer thread loads and preprocesses audio
        - Consumer (main thread) embeds audio while producer prepares next batch
        Ensures metadata and embeddings are written exactly like in the sequential version.

        Parameters
        ----------
        array_of_audios : np.array
            array with the audio samples to embed

        Returns
        -------
        list
            list with the embeddings for the audio samples
        """
        if len(array_of_audios.shape) == 1:
            array_of_audios = torch.tensor(array_of_audios).unsqueeze(0)
        windowed_audios = self._window_audio(array_of_audios)
        windowed_audios = torch.tensor(windowed_audios).unsqueeze(1)
        if not self.nr_parallel_workers:
            from multiprocessing import cpu_count

            available_workers = cpu_count() - 1
        else:
            available_workers = self.nr_parallel_workers
        task_queue = queue.Queue(
            maxsize=available_workers
        )  # small buffer to balance I/O vs compute

        # --- Producer: load + preprocess in background ---
        def producer():
            """Load and preprocess all audio samples in the background."""
            for idx, audio in enumerate(windowed_audios):
                try:
                    preprocessed = self.model.preprocess(audio)
                    task_queue.put((idx, preprocessed))

                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        "\nCuda device is out of memory. Your Vram doesn't seem to be "
                        "large enough for this process. Try setting the variable "
                        "`avoid_pipelined_gpu_inference` to `True`. That way data "
                        "will be processed in series instead of parallel which will "
                        "reduce memory requirements. If that also fails use `cpu` "
                        "instead of `cuda`."
                    )
                    os._exit(1)
                except Exception as e:
                    task_queue.put((idx, e))
            task_queue.put(None)  # sentinel = done

        threading.Thread(target=producer, daemon=True).start()

        # --- Consumer: embed + save metadata/embeddings ---
        embeddings = []
        with tqdm(
            total=len(windowed_audios),
            desc="processing audio",
            position=1,
            leave=False,
        ) as pbar:
            while True:
                item = task_queue.get()
                if item is None:
                    break

                idx, data = item
                if isinstance(data, Exception):
                    logger.warning(
                        f"Error preprocessing audio, skipping file.\nError: {data}"
                    )
                    pbar.update(1)
                    continue

                try:
                    embeddings.extend(self.get_embeddings_for_audio(data))

                except torch.cuda.OutOfMemoryError as e:
                    logger.exception(
                        "You're out of memory unfortunately. This can happend because you are "
                        "running a tensorflow model first and then a pytorch model and tensorflow "
                        "grabs all the VRAM and subsequently pytorch doesn't have enough. If this "
                        "is the case, simply rerunning bacpipe without the tensorflow model should "
                        "do the trick. If you simply don't have enough VRAM, you can reduce the global "
                        f"batch size in the settings file. Or just compute on the cpu. {str(e)}"
                    )
                except AttributeError as e:
                    logger.warning(
                        f"The results folder structure does not exist, therefore files can't be "
                        "saved. Please pass the keyword use_folder_structure=True."
                    )
                    pbar.update(1)
                except Exception as e:
                    logger.warning(
                        f"Error generating embeddings for audio, skipping file.\nError: {str(e)}"
                    )
                    pbar.update(1)
                    continue

                pbar.update(1)
        return embeddings

    def run_inference_pipeline_using_multithreading(self):
        """
        Generate embeddings for all files in a pipelined manner:
        - Producer thread loads and preprocesses audio
        - Consumer (main thread) embeds audio while producer prepares next batch
        Ensures metadata and embeddings are written exactly like in the sequential version.

        Generates embeddings for all files in the loader in a pipelined
        manner and saves the metadata and embeddings to disk.
        """
        if not self.nr_parallel_workers:
            from multiprocessing import cpu_count

            available_workers = cpu_count() - 1
        else:
            available_workers = self.nr_parallel_workers
        if self.loader.combination_already_exists:
            return
        task_queue = queue.Queue(
            maxsize=available_workers
        )  # small buffer to balance I/O vs compute

        # --- Producer: load + preprocess in background ---
        def producer():
            """Load and preprocess all audio files in the background."""
            for idx, file in enumerate(self.loader.files):
                try:
                    preprocessed = self.prepare_audio(file)
                    task_queue.put((idx, file, preprocessed))
                
                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        "\nCuda device is out of memory. Your Vram doesn't seem to be "
                        "large enough for this process. Try setting the variable "
                        "`avoid_pipelined_gpu_inference` to `True`. That way data "
                        "will be processed in series instead of parallel which will "
                        "reduce memory requirements. If that also fails use `cpu` "
                        "instead of `cuda`."
                    )
                    os._exit(1)
                except Exception as e:
                    if 'JIT compilation failed' in str(e):
                        error_str = (
                            "There was an error that is most likely caused by "
                            "using cuda and first running a pytorch and then a "
                            "tensorflow model. You have to run them in individual "
                            "sessions, then this error shouldn't occur."
                        )
                        logger.exception(
                            error_str
                        )
                        raise MemoryError(error_str)
                    task_queue.put((idx, file, e))
            task_queue.put(None)  # sentinel = done

        threading.Thread(target=producer, daemon=True).start()

        # --- Consumer: embed + save metadata/embeddings ---
        with tqdm(
            total=len(self.loader.files),
            desc="processing files",
            position=1,
            leave=False,
        ) as pbar:
            while True:
                item = task_queue.get()
                if item is None:
                    break

                idx, file, data = item
                if isinstance(data, Exception):
                    logger.warning(
                        f"Error preprocessing {file}, skipping file.\nError: {data}"
                    )
                    pbar.update(1)
                    continue

                try:
                    embeddings = self.get_embeddings_for_audio(data)

                    self.loader._write_audio_file_to_metadata(
                        file, self.model, embeddings, self.file_length
                    )
                    self.loader.save_embedding_file(file, embeddings)
                    if self.model.bool_classifier:
                        self.classifier.save_classifier_outputs(
                            self.loader, file
                        )
                except torch.cuda.OutOfMemoryError as e:
                    logger.exception(
                        "You're out of memory unfortunately. This can happend because you are "
                        "running a tensorflow model first and then a pytorch model and tensorflow "
                        "grabs all the VRAM and subsequently pytorch doesn't have enough. If this "
                        "is the case, simply rerunning bacpipe without the tensorflow model should "
                        "do the trick. If you simply don't have enough VRAM, you can reduce the global "
                        f"batch size in the settings file. Or just compute on the cpu. {str(e)}"
                    )
                    self.classifier.predictions = torch.tensor([])
                except AttributeError as e:
                    logger.warning(
                        f"The results folder structure does not exist, therefore files can't be "
                        "saved. Please pass the keyword use_folder_structure=True."
                    )
                    self.classifier.predictions = torch.tensor([])
                    pbar.update(1)
                except Exception as e:
                    logger.warning(
                        # This is not a process-ending exception because there are many reasons
                        # like corrupted files or other problems that can cause some files to 
                        # not process. But if users are running a long run, we do not want this
                        # run to fail because of minor problems.
                        f"Error generating embeddings for {file}, skipping file.\nError: {str(e)}"
                    )
                    # Do not carry a failed file's predictions over into the
                    # next file's classifier outputs.
                    self.classifier.predictions = torch.tensor([])
                    pbar.update(1)
                    continue

                pbar.update(1)

    def run_inference_pipeline_sequentially(self):
        """
        Generate embeddings for all files in the loader sequentially and
        save the metadata, the embeddings and classifier outputs to disk.
        """
        for idx, file in enumerate(
            tqdm(
                self.loader.files,
                desc="processing files",
                position=1,
                leave=False,
            )
        ):
            try:
                try:
                    embeddings = self.get_embeddings_from_model(file)
                except soundfile.LibsndfileError as e:
                    logger.warning(
                        f"\n Error loading audio, skipping file. \n"
                        f"Error: {str(e)}"
                    )
                    continue
            except Exception as e:
                logger.warning(
                    f"\n Error generating embeddings, skipping file. \n"
                    f"Error: {str(e)}"
                )
                # Do not carry a failed file's predictions over into the next
                # file's classifier outputs.
                self.classifier.predictions = torch.tensor([])
                continue

            self.loader._write_audio_file_to_metadata(
                file, self.model, embeddings, self.file_length
            )
            self.loader.save_embedding_file(file, embeddings)
            if self.model.bool_classifier:
                self.classifier.save_classifier_outputs(self.loader, file)

    def get_embeddings_from_model(self, sample):
        """
        Run full embedding generation pipeline, both for generating
        embeddings from audio data or generating dimensionality reductions
        from embedding data. Depending on that sample can be an embedding
        array or a audio file path.

        Parameters
        ----------
        sample : np.array or string-like
            embedding array of path to audio file

        Returns
        -------
        np.array
            embeddings
        """
        start = time.time()
        if self.dim_reduction_model:
            embeds = self.get_reduced_dimensionality_embeddings(sample)
        else:
            if not isinstance(sample, Path):
                sample = Path(sample)
                if not hasattr(self, 'audio_suffixes'):
                    self.audio_suffixes = bacpipe.settings.audio_suffixes
                if not sample.suffix in self.audio_suffixes:
                    error = (
                        "\nThe provided path does not lead to a supported audio file with the ending"
                        f" {self.audio_suffixes}. Please check again that you provided the correct"
                        " path."
                    )
                    logger.exception(error)
                    raise AssertionError(error)
            sample = self.prepare_audio(sample)
            embeds = self.get_embeddings_for_audio(sample)

        logger.debug(
            f"{self.model_name} embeddings have shape: {embeds.shape}"
        )
        logger.info(
            f"{self.model_name} inference took {time.time()-start:.2f}s."
        )
        return embeds


class Classifier:
    """
    Handle all tasks surrounding classification: generating predictions from
    embeddings, collecting them into arrays and creating dataframes and
    annotation tables from them.
    """

    def __init__(
        self,
        model,
        model_name,
        audio_dir,
        main_results_dir,
        classifier_threshold,
        use_folder_structure=True,
        save_raven_tables=False,
        **kwargs,
    ):
        """
        Class to handle all tasks surrounding classification. Both generating
        the classifications from embeddings, as well as managing them, collecting
        them in arrays and creating dataframes and annotation tables from them.

        Parameters
        ----------
        model : Model object
            has attributes for all the model characteristics like
            sample rate, segment length etc. as well as the methods
            to run the model
        model_name : str
            name of the model
        classifier_threshold : float, optional
            Value under which class predictions are discarded, by default None
        audio_dir : str
            path to the directory containing the audio files
        main_results_dir : str
            path to the main results directory
        use_folder_structure : bool, optional
            if True, the results are saved in the folder structure,
            by default True
        save_raven_tables : bool, optional
            if True, Raven annotation tables are saved, by default False
        """
        self.model = model
        self.model_name = model_name
        self.classifier_threshold = classifier_threshold
        self.save_raven_tables = save_raven_tables
        self.max_labels_per_timestamp = kwargs.get(
            "max_labels_per_timestamp", bacpipe.settings.max_labels_per_timestamp
        )
        if use_folder_structure:
            from bacpipe.embedding_evaluation.label_embeddings import (
                make_set_paths_func,
            )

            self.paths = make_set_paths_func(
                audio_dir,
                main_results_dir,
                evaluations_dir=kwargs.get("evaluations_dir"),
            )(model_name)

        self.predictions = torch.tensor([])

        if kwargs.get("only_embed_annotations"):
            self.only_embed_annotations = True
            if not use_folder_structure:
                from bacpipe.embedding_evaluation.label_embeddings import (
                    load_labels_and_build_dict
                )
                self.df = load_labels_and_build_dict(
                    paths=None,
                    annotations_filename=kwargs.get("annotations_filename"),
                    audio_dir=audio_dir,
                    bool_filter_labels=False
                )
            else:
                from bacpipe.embedding_evaluation.label_embeddings import (
                    load_labels_and_build_dict,
                    assign_global_get_paths_function,
                    get_paths,
                )

                assign_global_get_paths_function(audio_dir, **kwargs)
                paths = get_paths(self.model_name)
                self.df = load_labels_and_build_dict(
                    paths,
                    kwargs.get("annotations_filename"),
                    audio_dir,
                    bool_filter_labels=False,
                )
            self.start_timestamps = self.df.start.values
            self.end_timestamps = self.df.end.values

    @staticmethod
    def filter_top_k_classifications(
        probabilities, class_names, threshold, max_labels_per_timestamp=None
    ):
        """
        Generate a dictionary with the top k classes. By limiting the class number to
        k, it prevents from this step taking too long but has the benefit of generating
        a dicitonary which can be saved as a .json file to quickly get a overview of
        species that are well represented within an audio file.

        Parameters
        ----------
        probabilities : np.array
            Probabilities for each class
        class_names : list
            class names
        threshold : float
            values to threshold probabilities with
        max_labels_per_timestamp : int, optional
            number of labels to keep per timestamp, by default None. If None,
            the value from ``bacpipe.settings.max_labels_per_timestamp``
            is used.

        Returns
        -------
        dict
            dictionary of top k classes with time bin indices exceeding threshold
        """
        if max_labels_per_timestamp is None:
            max_labels_per_timestamp = (
                bacpipe.settings.max_labels_per_timestamp
            )
        if not max_labels_per_timestamp is None:
            k = max_labels_per_timestamp
            
        top_k_indices = np.argsort(np.array(probabilities), axis=0)[-k:][::-1]
        top_k_probs = np.sort(probabilities, axis=0)[-k:][::-1]
        
        top_k_indices[top_k_probs <= threshold] = -1
        unique_classes = np.unique(top_k_indices)
        
        cls_results = {
            class_names[one_class]: {
                "time_bins_exceeding_threshold": np.where(
                    top_k_indices==one_class
                    )[-1].tolist(),
                "classifier_predictions": np.array(
                    top_k_probs[top_k_indices==one_class]
                ).tolist(),
            }
            for one_class in unique_classes
            if one_class >= 0
        }
        return cls_results

    @staticmethod
    def make_classification_dict(
        probabilities, classes, threshold, max_labels_per_timestamp=None
    ):
        """
        Make a classification dictionary for one audio file with the top k
        classifications and a head with general information about the file.

        Parameters
        ----------
        probabilities : np.array
            probabilities for each class
        classes : list
            class names
        threshold : float
            value to threshold the probabilities with
        max_labels_per_timestamp : int, optional
            number of labels to keep per timestamp, by default None. If None,
            the value from ``bacpipe.settings.max_labels_per_timestamp``
            is used.

        Returns
        -------
        dict
            dictionary with the classification results for the audio file
        """
        if probabilities.shape[0] != len(classes):
            probabilities = probabilities.swapaxes(0, 1)

        cls_results = Classifier.filter_top_k_classifications(
            probabilities,
            classes,
            threshold,
            max_labels_per_timestamp=max_labels_per_timestamp,
        )

        cls_results["head"] = {
            "Time bins in this file": probabilities.shape[-1],
            "Threshold for classifier predictions": threshold,
        }
        return cls_results

    def classify(self, embeddings):
        """
        Classify embeddings and collect the predictions.

        Parameters
        ----------
        embeddings : torch.Tensor or np.array
            embeddings to be classified
        """
        if not isinstance(embeddings, torch.Tensor):
            clfier_output = self.model.classifier_predictions(
                torch.tensor(np.array(embeddings))
            )
        else:
            clfier_output = self.model.classifier_predictions(embeddings)

        if self.model.device == "cuda" and isinstance(
            clfier_output, torch.Tensor
        ):
            self.predictions = self.predictions.cuda()
            clfier_output = clfier_output.cuda()

        if isinstance(clfier_output, torch.Tensor):
            self.predictions = torch.cat(
                [self.predictions, clfier_output.clone().detach()]
            )
        else:
            self.predictions = torch.cat(
                [self.predictions, torch.Tensor(clfier_output)]
            )

    def _fill_dataframe_with_classiefier_results(self, fileloader_obj, file):
        """
        Append or create a dataframe and fill it with the results from the
        classifier to later be saved as a csv file.

        Parameters
        ----------
        fileloader_obj : bacpipe.Loader object
            All paths and metadata of embeddings creation run
        file : pathlike
            audio file path
        """
        classifier_annotations = pd.DataFrame()

        maxes = torch.max(self.predictions, dim=1)

        if hasattr(self, "only_embed_annotations") and getattr(
            self, "only_embed_annotations"
        ):
            from bacpipe import Loader

            df = Loader.filter_df_by_file(
                fileloader_obj.audio_dir, self.df, file
            )
            starts = (
                df.start.values
            )  # self.start_timestamps[self.df.audiofilename == str(file.relative_to(fileloader_obj.audio_dir))]
            ends = (
                df.end.values
            )  # self.end_timestamps[self.df.audiofilename == str(file.relative_to(fileloader_obj.audio_dir))]
            starts, ends = list(set(starts)), list(set(ends))
            starts.sort(), ends.sort()
            starts, ends = np.array(starts), np.array(ends)
            classifier_annotations["start"] = starts[
                maxes.values > self.classifier_threshold
            ]
            classifier_annotations["end"] = ends[
                maxes.values > self.classifier_threshold
            ]
        else:
            time_bins = np.arange(self.predictions.shape[0])
            self.start_timestamps = time_bins * (
                self.model.segment_length / self.model.sr
            )
            self.end_timestamps = self.start_timestamps + (
                self.model.segment_length / self.model.sr
            )
            classifier_annotations["start"] = self.start_timestamps[
                maxes.values > self.classifier_threshold
            ]
            classifier_annotations["end"] = self.end_timestamps[
                maxes.values > self.classifier_threshold
            ]
        classifier_annotations["audiofilename"] = str(
            file.relative_to(fileloader_obj.audio_dir).as_posix()
        )
        # Index into the classes array with an explicit 1-D
        # numpy index array. A single-element torch index makes numpy return
        # a plain ``str`` scalar (which has no ``.tolist()``), crashing on any
        # file that has exactly one bin exceeding the threshold.
        detected_bin_idxs = np.atleast_1d(
            np.array(maxes.indices[maxes.values > self.classifier_threshold])
        ).astype(int)
        classifier_annotations["label:default_classifier"] = np.array(
            self.model.classes
        )[detected_bin_idxs].tolist()

        classifier_annotations["label:confidence"] = np.array(
            maxes.values[maxes.values > self.classifier_threshold].tolist()
        )

        if not hasattr(self, "cumulative_annotations"):
            if fileloader_obj.continue_incomplete_run:
                self._load_existing_clfier_outputs(
                    fileloader_obj, classifier_annotations
                )
            else:
                self.cumulative_annotations = classifier_annotations
        else:
            self.cumulative_annotations = pd.concat(
                [self.cumulative_annotations, classifier_annotations],
                ignore_index=True,
            )

    def _load_existing_clfier_outputs(
        self, fileloader_obj: bacpipe.Loader, clfier_annotations=None
    ):
        """
        Load the existing classifier outputs from the predictions file and
        combine them with the new classifier annotations.

        Parameters
        ----------
        fileloader_obj : bacpipe.Loader object
            all paths and metadata of the embedding creation run
        clfier_annotations : pandas.DataFrame, optional
            dataframe with the new classifier annotations, by default None
        """
        df_dict = {
            "start": [],
            "end": [],
            "audiofilename": [],
            "label:default_classifier": [],
        }

        df = fileloader_obj.predictions(return_type="dataframe")
        if isinstance(df, pd.DataFrame):
            df_dict = df[["audiofilename", "start", "end"]].copy()
            if len(df) > 0:
                # Select the species columns by name rather than by position
                # so that a leading index column (e.g. "Unnamed: 0") in
                # prediction files written by older bacpipe versions cannot
                # shift the species columns and mis-assign the species that
                # belongs to a given embedding row.
                species_cols = [
                    col
                    for col in df.columns
                    if col not in {
                        "audiofilename",
                        "start",
                        "end",
                        "simultaneous_labels",
                    }
                    and not str(col).startswith("Unnamed")
                ]
                if len(species_cols) > 0:
                    species_df = df[species_cols]
                    maxes = np.argmax(np.array(species_df), axis=1)
                    df_dict["label:default_classifier"] = [
                        species_df.columns[i] for i in maxes
                    ]

        self.cumulative_annotations = pd.DataFrame(df_dict)
        self.cumulative_annotations = pd.concat(
            [self.cumulative_annotations, clfier_annotations],
            ignore_index=True,
        )

    def save_annotation_table(self, loader_obj: bacpipe.Loader, **kwargs):
        """
        Save the cumulative classifier annotations as a csv file.

        Parameters
        ----------
        loader_obj : bacpipe.Loader object
            all paths and metadata of the embedding creation run
        """
        self.paths.preds_path.mkdir(exist_ok=True, parents=True)
        loader_obj.get_annotations_parquet(
            starts=self.start_timestamps, ends=self.end_timestamps, **kwargs
        )
        save_path = (
            self.paths.preds_path
            / f"{loader_obj.model_name}_classifier_annotations.csv"
        )
        self.cumulative_annotations.to_csv(save_path, index=False)

    def save_classifier_outputs(self, fileloader_obj, file):
        """
        Save the classification results for a single audio file as a json
        file and optionally as a Raven annotation table.

        Parameters
        ----------
        fileloader_obj : bacpipe.Loader object
            all paths and metadata of the embedding creation run
        file : pathlib.Path
            path to the audio file
        """
        if len(self.predictions.shape) == 1:
            self.predictions = self.predictions.unsqueeze(0)
        elif self.predictions.shape[-1] != len(self.model.classes):
            self.predictions = self.predictions.swapaxes(0, 1)

        relative_parent_path = (
            Path(file).relative_to(fileloader_obj.audio_dir).parent
        )
        results_path = self.paths.preds_path.joinpath(
            "original_classifier_outputs"
        ).joinpath(relative_parent_path)
        results_path.mkdir(exist_ok=True, parents=True)
        file_dest = results_path.joinpath(file.stem + "_" + self.model_name)
        file_dest = str(file_dest) + ".json"

        # if self.model.only_embed_annotations: #annotation file exists
        #     np.save(file_dest.replace('.json', '.npy'), self.predictions)

        try:
            self._fill_dataframe_with_classiefier_results(fileloader_obj, file)

            cls_results = self.make_classification_dict(
                self.predictions,
                self.model.classes,
                self.classifier_threshold,
                max_labels_per_timestamp=self.max_labels_per_timestamp,
            )

            with open(file_dest, "w") as f:
                json.dump(cls_results, f, indent=2)

            if self.save_raven_tables:
                self.save_Raven_table(file, relative_parent_path)
        except Exception:
            # If anything goes wrong while saving this file, drop its
            # predictions so they are NOT carried over into the next file's
            # classifier outputs. 
            self.predictions = torch.tensor([])
            raise

        self.predictions = torch.tensor([])

    def save_Raven_table(self, file, relative_parent_path):
        """
        Save the classifier predictions of a single audio file as a Raven
        annotation table.

        Parameters
        ----------
        file : pathlib.Path
            path to the audio file
        relative_parent_path : pathlib.Path
            path of the parent folder relative to the audio directory
        """
        raven_results_path = self.paths.preds_path.joinpath(
            "raven_tables"
        ).joinpath(relative_parent_path)
        raven_results_path.mkdir(exist_ok=True, parents=True)
        raven_file_dest = raven_results_path.joinpath(
            file.stem + "_" + self.model_name
        )
        raven_file_dest = str(raven_file_dest) + ".selection.table.txt"
        
        if not self.max_labels_per_timestamp is None:
            k = self.max_labels_per_timestamp
            
        top_k_indices = np.argsort(np.array(self.predictions), axis=1)[:,-k:][:, ::-1]
        top_k_probs = np.sort(np.array(self.predictions), axis=1)[:,-k:][:, ::-1]
        
        top_k_indices[top_k_probs <= self.classifier_threshold] = 0

        timestamps = np.where(top_k_indices)[0]
        species = top_k_indices[top_k_indices>0].flatten()
        probs = [
            np.array(self.predictions)[ts, sp]
            for ts, sp in zip(timestamps, species)
        ]
        specs = [self.model.classes[sp] for sp in species]

        df = pd.DataFrame()
        df["label:species"] = specs
        df["start"] = self.start_timestamps[timestamps]
        df["end"] = self.end_timestamps[timestamps]
        from bacpipe.embedding_evaluation.label_embeddings import (
            create_Raven_annotation_table,
        )

        raven_df = create_Raven_annotation_table(
            df, "species", high_freq=self.model.sr * np.array(probs)
        )
        raven_df["Confidence"] = probs
        raven_df["Begin Path"] = relative_parent_path / (
            file.stem + file.suffix
        )
        raven_df["File Offset (s)"] = df.start
        if len(raven_df) > 0:
            # only save table if there are predictions
            raven_df.to_csv(raven_file_dest, sep="\t", index=False)

    def run_default_classifier(self, loader):
        """
        Run the pretrained classifier on all embeddings, save the classifier
        outputs for every file and save the cumulative annotation table.

        Parameters
        ----------
        loader : bacpipe.Loader object
            all paths and metadata of the embedding creation run
        """
        all_embeds = loader.embeddings()
        for f_name, embeddings in tqdm(
            all_embeds.items(),
            desc="Running pretrained classifier",
            total=len(all_embeds),
        ):

            if not isinstance(embeddings, torch.Tensor):
                clfier_output = self.model.classifier_predictions(
                    torch.tensor(np.array(embeddings))
                )
            else:
                clfier_output = self.model.classifier_predictions(embeddings)

            if isinstance(clfier_output, torch.Tensor):
                self.predictions = torch.cat(
                    [self.predictions, clfier_output.clone().detach()]
                )
            else:
                self.predictions = torch.cat(
                    [self.predictions, torch.Tensor(clfier_output)]
                )

            f_name_stem = f_name.split(f"_{self.model_name}")[0]
            try:
                audiofile = [
                    f
                    for f in loader.metadata_dict["files"]["audio_files"]
                    if f_name_stem in f
                ][0]
            except:
                raise AssertionError(
                    f"{f_name} has no corresponding audio file. "
                    "Something is wrong in the metadata file. "
                    "Please either inspect manually or rerun bacpipe "
                    "and remove the existing embeddings folder."
                )

            self.save_classifier_outputs(loader, loader.audio_dir / audiofile)

        self.save_annotation_table(loader)

        if loader.model_name in bacpipe.TF_MODELS:
            import tensorflow as tf

            tf.keras.backend.clear_session()
