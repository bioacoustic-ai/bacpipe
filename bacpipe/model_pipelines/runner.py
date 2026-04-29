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
from bacpipe.core.experiment_manager import replace_default_kwargs_with_user_kwargs
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
        """
        self.file_length = {}
        self.loader = loader

        self.dim_reduction_model = dim_reduction_model
        if dim_reduction_model:
            self.dim_reduction_model = True
            self.model_name = dim_reduction_model
        else:
            self.model_name = model_name
            
        kwargs = replace_default_kwargs_with_user_kwargs(
            ['audio_dir', 'models', 'dim_reduction_model'], kwargs=kwargs
        )
        self.nr_parallel_workers = kwargs.get('nr_parallel_workers')
        
        self._init_model(
            dim_reduction_model=dim_reduction_model, 
            CustomModel=CustomModel,
            **kwargs
            )
        super().__init__(
            model=self.model, 
            audio_dir=loader.audio_dir if loader else None, 
            **kwargs
            )
        if self.model.bool_classifier:
            self.classifier = Classifier(
                self.model, 
                model_name, 
                audio_dir=loader.audio_dir if loader else None, 
                use_folder_structure=loader.use_folder_structure if loader else False,
                **kwargs
                )

    def _init_model(self, CustomModel=None, **kwargs):
        """
        Load model specific module, instantiate model and allocate device for model.
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
        if "tensorflow" in str(type(audio)):
            import tensorflow as tf

            return (
                tf.data.Dataset
                .from_tensor_slices(audio)
                .batch(self.model.batch_size)
                )
            
        elif "torch" in str(type(audio)):

            return torch.utils.data.DataLoader(
                audio, batch_size=self.model.batch_size, shuffle=False
            )

    def batch_inference(self, batched_samples, callback=None):
        if self.model_name in bacpipe.TF_MODELS:
            import tensorflow
        
        embeds = []
        total_batches = len(batched_samples)

        for idx, batch in enumerate(
            tqdm(batched_samples, desc=" processing batches", position=0, leave=False)
        ):
            with torch.no_grad():
                if (
                    self.model.device == "cuda" 
                    and hasattr(batch, 'cuda')
                    ):
                    batch = batch.cuda()
                embeddings = self.model(batch)
                if self.model.bool_classifier:
                    self.classifier.classify(embeddings)

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
            if len(return_embeds.shape) > 2:
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
        if self.loader.metadata_dict['nr_embeds_total'] > 300_000:
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
                / int(np.mean(self.loader.metadata_dict['files']['nr_embeds_per_file']))
                )
            sample_files = (
                np.random.permutation(self.loader.files)[:sample_file_size]
                )
        else:
            use_sample_of_files = False
            sample_files = self.loader.files
        
        embeddings_list = []
        for file in tqdm(
            sample_files, 
            desc="loading files", 
            position=1, 
            leave=False
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
                leave=False
                ):
                dim_reduced_embeddings_list.append(
                    self.model.model.transform(
                        self.loader.read_embedding_file(file)
                        )
                    )

            dim_reduced_embeddings = np.concatenate(dim_reduced_embeddings_list, axis=0)
        else:
            dim_reduced_embeddings = self.get_embeddings_from_model(embeddings)
        self.loader.save_embedding_file(file, dim_reduced_embeddings)

    def embeddings_using_multithreading(self, array_of_audios):
        """
        Generate embeddings for all files in a pipelined manner:
        - Producer thread loads and preprocesses audio
        - Consumer (main thread) embeds audio while producer prepares next batch
        Ensures metadata and embeddings are written exactly like in the sequential version.

        Parameters
        ----------
        fileloader_obj : Loader object
            contains all metadata of a model specific embedding creation session

        Returns
        -------
        Loader object
            updated object with metadata on embedding creation session
        """
        if len(array_of_audios.shape) == 1:
            array_of_audios = torch.tensor(array_of_audios).unsqueeze(0)
        windowed_audios = self._window_audio(array_of_audios)
        windowed_audios = torch.tensor(windowed_audios).unsqueeze(1)
        if not self.nr_parallel_workers:
            from multiprocessing import cpu_count
            available_workers = cpu_count() -1
        else:
            available_workers = self.nr_parallel_workers
        task_queue = queue.Queue(maxsize=available_workers)  # small buffer to balance I/O vs compute
        # --- Producer: load + preprocess in background ---
        def producer():
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
                        f"batch size in the settings file. Or just compute on the cpu. {e}"
                    )
                except AttributeError as e:
                    logger.warning(
                        f"The results folder structure does not exist, therefore files can't be "
                        "saved. Please pass the keyword use_folder_structure=True."
                    )
                    pbar.update(1)
                except Exception as e:
                    logger.warning(
                        f"Error generating embeddings for audio, skipping file.\nError: {e}"
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

        Parameters
        ----------
        fileloader_obj : Loader object
            contains all metadata of a model specific embedding creation session

        Returns
        -------
        Loader object
            updated object with metadata on embedding creation session
        """
        if not self.nr_parallel_workers:
            from multiprocessing import cpu_count
            available_workers = cpu_count() -1
        else:
            available_workers = self.nr_parallel_workers
        if self.loader.combination_already_exists:
            return
        task_queue = queue.Queue(maxsize=available_workers)  # small buffer to balance I/O vs compute
        # --- Producer: load + preprocess in background ---
        def producer():
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
                        self.classifier.save_classifier_outputs(self.loader, file)
                except torch.cuda.OutOfMemoryError as e:
                    logger.exception(
                        "You're out of memory unfortunately. This can happend because you are "
                        "running a tensorflow model first and then a pytorch model and tensorflow "
                        "grabs all the VRAM and subsequently pytorch doesn't have enough. If this "
                        "is the case, simply rerunning bacpipe without the tensorflow model should "
                        "do the trick. If you simply don't have enough VRAM, you can reduce the global "
                        f"batch size in the settings file. Or just compute on the cpu. {e}"
                    )
                except AttributeError as e:
                    logger.warning(
                        f"The results folder structure does not exist, therefore files can't be "
                        "saved. Please pass the keyword use_folder_structure=True."
                    )
                    pbar.update(1)
                except Exception as e:
                    logger.warning(
                        f"Error generating embeddings for {file}, skipping file.\nError: {e}"
                    )
                    pbar.update(1)
                    continue

                pbar.update(1)

    def run_inference_pipeline_sequentially(self):
        for idx, file in enumerate(
            tqdm(self.loader.files, desc="processing files", position=1, leave=False)
        ):
            try:
                try:
                    embeddings = self.get_embeddings_from_model(file)
                except soundfile.LibsndfileError as e:
                    logger.warning(
                        f"\n Error loading audio, skipping file. \n"
                        f"Error: {e}"
                    )
                    continue
            except Exception as e:
                logger.warning(
                    f"\n Error generating embeddings, skipping file. \n"
                    f"Error: {e}"
                )
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
                # TODO might not know audio_suffixes
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

        logger.debug(f"{self.model_name} embeddings have shape: {embeds.shape}")
        logger.info(f"{self.model_name} inference took {time.time()-start:.2f}s.")
        return embeds

class Classifier:
    def __init__(
        self, 
        model, 
        model_name, 
        audio_dir, 
        main_results_dir, 
        classifier_threshold, 
        use_folder_structure=True, 
        save_raven_tables=False,
        **kwargs
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
        """
        self.model = model
        self.model_name = model_name
        self.classifier_threshold = classifier_threshold
        self.save_raven_tables = save_raven_tables
        if use_folder_structure:
            from bacpipe.embedding_evaluation.label_embeddings import make_set_paths_func
            self.paths = make_set_paths_func(audio_dir, main_results_dir)(model_name)
        
        self.predictions = torch.tensor([])
        
    @staticmethod
    def filter_top_k_classifications(probabilities, class_names,
                                     class_indices, class_time_bins, 
                                     k=50):
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
        class_indices : np.array
            class indices exceeding the threshold
        class_time_bins : np.array
            time bin indices exceeding the threshold
        k : int, optional
            number of classes to save in the dict. keep this below 100
            otherwise the operation will start slowing the process down
            a lot, by default 50

        Returns
        -------
        dict
            dictionary of top k classes with time bin indices exceeding threshold
        """
        classes, class_counts = np.unique(class_indices, 
                                          return_counts=True)
        
        cls_dict = {k: v for k, v in zip(classes, class_counts)}
        cls_dict = dict(sorted(cls_dict.items(), key=lambda x: x[1], 
                               reverse=True))
        top_k_cls = {k: v for i, (k, v) 
                     in enumerate(cls_dict.items()) 
                     if i < k}
                
        cls_results = {
            class_names[cls]: {
                "time_bins_exceeding_threshold": class_time_bins[
                    class_indices == cls
                    ].tolist(),
                "classifier_predictions": np.array(
                    probabilities[class_indices[class_indices == cls], 
                                  class_time_bins[class_indices == cls]]
                ).tolist(),
            }
            for cls in top_k_cls.keys()
        }
        return cls_results

    @staticmethod
    def make_classification_dict(probabilities, classes, threshold):
        if probabilities.shape[0] != len(classes):
            probabilities = probabilities.swapaxes(0, 1)

        cls_idx, tmp_idx = np.where(probabilities > threshold)

        cls_results = Classifier.filter_top_k_classifications(probabilities, 
                                                            classes,
                                                            cls_idx, 
                                                            tmp_idx)

        cls_results["head"] = {
            "Time bins in this file": probabilities.shape[-1],
            "Threshold for classifier predictions": threshold,
        }
        return cls_results
    
    def classify(self, embeddings):
        try:
            clfier_output = self.model.classifier_predictions(torch.tensor(embeddings))
        except:
            clfier_output = self.model.classifier_predictions(embeddings)
            
        if self.model.device == "cuda" and isinstance(clfier_output, torch.Tensor):
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
        # outputs_exceeding_thresh = self.predictions[
        #     maxes.values > self.classifier_threshold
        # ]
        
        active_time_bins = np.arange(
            self.predictions.shape[0]
            )[maxes.values > self.classifier_threshold]
        
        classifier_annotations["start"] = active_time_bins * (
            self.model.segment_length / self.model.sr
        )
        classifier_annotations["end"] = classifier_annotations["start"] + (
            self.model.segment_length / self.model.sr
        )
        classifier_annotations["audiofilename"] = str(
            file.relative_to(fileloader_obj.audio_dir)
        )
        classifier_annotations["label:default_classifier"] = np.array(
            self.model.classes
        )[maxes.indices[maxes.values > self.classifier_threshold]].tolist()

        if not hasattr(self, "cumulative_annotations"):
            if fileloader_obj.continue_incomplete_run:
                self._load_existing_clfier_outputs(fileloader_obj, 
                                                  classifier_annotations)
            else:
                self.cumulative_annotations = classifier_annotations
        else:
            self.cumulative_annotations = pd.concat(
                [self.cumulative_annotations, classifier_annotations], 
                ignore_index=True
            )

    def _load_existing_clfier_outputs(self, fileloader_obj: bacpipe.Loader, clfier_annotations=None):
        df_dict = {
            'start': [],
            'end': [],
            'audiofilename': [],
            'label:default_classifier': []
            }
        
        df = fileloader_obj.predictions(return_type='dataframe')
        df_dict = df[['audiofilename', 'start', 'end']]
        if len(df) > 0:
            cols = df.iloc[:, 4:].columns
            maxes = np.argmax(np.array(df.iloc[:, 4:]), axis=1)
            highest_prob_species = [cols[i] for i in maxes]

            df.iloc[:, 4:].max(axis=1)
            df_dict['label:default_classifier'] = highest_prob_species

        self.cumulative_annotations = pd.DataFrame(df_dict)
        self.cumulative_annotations = pd.concat(
                [self.cumulative_annotations, clfier_annotations], 
                ignore_index=True
            )
        
    def save_annotation_table(self, loader_obj: bacpipe.Loader):
        self.paths.preds_path.mkdir(exist_ok=True, parents=True)
        loader_obj.get_annotations_parquet()
        save_path = (
            self.paths.preds_path 
            / f"{loader_obj.model_name}_classifier_annotations.csv"
            )
        self.cumulative_annotations.to_csv(save_path, index=False)
        
    def save_classifier_outputs(self, fileloader_obj, file):
        if len(self.predictions.shape) == 1:
            self.predictions = self.predictions.unsqueeze(0)
        elif self.predictions.shape[-1] != len(self.model.classes):
            self.predictions = self.predictions.swapaxes(0, 1)
            
            
        relative_parent_path = Path(file).relative_to(fileloader_obj.audio_dir).parent
        results_path = self.paths.preds_path.joinpath(
            "original_classifier_outputs"
        ).joinpath(relative_parent_path)
        results_path.mkdir(exist_ok=True, parents=True)
        file_dest = results_path.joinpath(file.stem + "_" + self.model_name)
        file_dest = str(file_dest) + ".json"


        if self.model.only_embed_annotations: #annotation file exists
            np.save(file_dest.replace('.json', '.npy'), self.predictions)
        
        self._fill_dataframe_with_classiefier_results(fileloader_obj, file)
        
        cls_results = self.make_classification_dict(
            self.predictions, self.model.classes, self.classifier_threshold
        )

        with open(file_dest, "w") as f:
            json.dump(cls_results, f, indent=2)
            
        if self.save_raven_tables:
            self.save_Raven_table(file, relative_parent_path)
            
        self.predictions = torch.tensor([])
        
    def save_Raven_table(self, file, relative_parent_path):
        raven_results_path = self.paths.preds_path.joinpath(
                "raven_tables"
            ).joinpath(relative_parent_path)
        raven_results_path.mkdir(exist_ok=True, parents=True)
        raven_file_dest = raven_results_path.joinpath(file.stem + "_" + self.model_name)
        raven_file_dest = str(raven_file_dest) + ".selection.table.txt"
        
        timestamps, species = np.where(
            self.predictions > self.classifier_threshold
            )
        probs = [
            np.array(self.predictions)[ts, sp] 
            for ts, sp in zip(timestamps, species)
            ]
        specs = [self.model.classes[sp] for sp in species]
        
        df = pd.DataFrame()
        df['label:species'] = specs
        df['start'] = timestamps * (self.model.segment_length / self.model.sr)
        df['end'] = df.start + (self.model.segment_length / self.model.sr)
        from bacpipe.embedding_evaluation.label_embeddings import create_Raven_annotation_table
        raven_df = create_Raven_annotation_table(df, 'species')
        raven_df['Confidence'] = probs
        raven_df["Begin Path"] = relative_parent_path / (file.stem + file.suffix)
        raven_df["File Offset (s)"] = df.start
        raven_df.to_csv(raven_file_dest, sep='\t', index=False)
    
    def run_default_classifier(self, loader):
        all_embeds = loader.embeddings()
        for f_name, embeddings in tqdm(
            all_embeds.items(),
            desc='Running pretrained classifier',
            total=len(all_embeds)
            ):        
            
            try:
                clfier_output = self.model.classifier_predictions(torch.tensor(embeddings))
            except:
                clfier_output = self.model.classifier_predictions(embeddings)

            if isinstance(clfier_output, torch.Tensor):
                self.predictions = torch.cat(
                    [self.predictions, clfier_output.clone().detach()]
                )
            else:
                self.predictions = torch.cat(
                    [self.predictions, torch.Tensor(clfier_output)]
                )

            f_name_stem = f_name.split(f'_{self.model_name}')[0]
            try:
                audiofile = [
                    f for f 
                    in loader.metadata_dict['files']['audio_files']
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