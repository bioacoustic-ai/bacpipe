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
logger = logging.getLogger("bacpipe")


class Embedder(AudioHandler):
    def __init__(
        self,
        model_name,
        loader, 
        dim_reduction_model=False,
        **kwargs,
    ):
        """
        This class defines all the entry points to generate embedding files. 
        Parameters are kept minimal, to accomodate as many cases as possible.
        At the end if instantiation, the selected model is loaded and the 
        model is associated with the device specified.

        Parameters
        ----------
        model_name : str
            name of selected embedding model
        loader : Loader object
            Object that has all the necessary path information and methods
            to load and save all the processed data
        dim_reduction_model : bool, optional
            Can be bool or the string corresponding to the 
            dimensionality reduction model, by default False
        testing : bool, optional
            _description_, by default False
        """
        self.file_length = {}
        self.loader = loader

        self.dim_reduction_model = dim_reduction_model
        if dim_reduction_model:
            self.dim_reduction_model = True
            self.model_name = dim_reduction_model
        else:
            self.model_name = model_name
        self._init_model(dim_reduction_model=dim_reduction_model, **kwargs)
        super().__init__(model=self.model, **kwargs)
        if self.model.bool_classifier:
            self.classifier = Classifier(
                self.model, model_name, **kwargs
                )

    def _init_model(self, **kwargs):
        """
        Load model specific module, instantiate model and allocate device for model.
        """
        if self.dim_reduction_model:
            module = importlib.import_module(
                f"bacpipe.model_pipelines.dimensionality_reduction.{self.model_name}"
            )
        else:
            module = importlib.import_module(
                f"bacpipe.model_pipelines.feature_extractors.{self.model_name}"
            )
        self.model = module.Model(model_name=self.model_name, **kwargs)
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
            return_embeds = tf.concat(embeds, axis=0).numpy().squeeze()
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
        for idx, file in enumerate(
            tqdm(self.loader.files, desc="processing files", position=1, leave=False)
        ):
            if idx == 0:
                embeddings = self.loader.read_embedding_file(file)
            else:
                embeddings = np.concatenate(
                    [embeddings, self.loader.read_embedding_file(file)]
                )

        dim_reduced_embeddings = self.get_embeddings_from_model(embeddings)
        self.loader.save_embedding_file(file, dim_reduced_embeddings)

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
        task_queue = queue.Queue(maxsize=4)  # small buffer to balance I/O vs compute

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
                except Exception as e:
                    logger.warning(
                        f"Error generating embeddings for {file}, skipping file.\nError: {e}"
                    )
                    pbar.update(1)
                    continue

                pbar.update(1)

    def run_inference_pipeline_sequentially(self):
        if self.model_name in bacpipe.TF_MODELS:
            import tensorflow as tf
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
                except tf.errors.ResourceExhaustedError: # TODO this needs fixing
                                            
                    logger.error(
                        "\nGPU device is out of memory. Your Vram doesn't seem to be "
                        "large enough for this process. This could be down to the "
                        "size of the audio files. Use `cpu` instead of `cuda`."
                    )
                    os._exit(1) 
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
        clfier_output = self.model.classifier_predictions(embeddings)
            
        if self.model.device == "cuda" and isinstance(clfier_output, torch.Tensor):
            self.predictions = self.predictions.cuda()

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
        
        maxes = torch.max(self.predictions, dim=0)
        outputs_exceeding_thresh = self.predictions[:,
            maxes.values > self.classifier_threshold
        ]
        
        active_time_bins = np.arange(
            self.predictions.shape[-1]
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

    def _load_existing_clfier_outputs(self, fileloader_obj, clfier_annotations):
        clfier_dir = Path(
            self.paths.class_path / 'original_classifier_outputs'
            )
        existing_clfier_outputs = list(clfier_dir.rglob('*.json'))
        existing_clfier_outputs.sort()
        
        seg_len = self.model.segment_length / self.model.sr
        
        relative_audio = np.array(
            # we omit the last item assuming that it's just been processed
            # and corresponds to the clfier_annotations contents
            [
                f.split('.')
                for f in 
                fileloader_obj.metadata_dict['files']['audio_files'][:-1]
                ]
            )
        relative_audio_stems = relative_audio[:, 0]
        relative_audio_suffixes = relative_audio[:, 1]
        
        
        df_dict = {
            'start': [],
            'end': [],
            'audiofilename': [],
            'label:default_classifier': []
            }
        for file in existing_clfier_outputs:
            corresponding_audio_file_bool = (
                relative_audio_stems==str(
                    file.relative_to(clfier_dir)
                    ).replace(f'_{fileloader_obj.model_name}.json', '')
            )
            with open(file, 'r') as f:
                outputs = json.load(f)
            outputs.pop('head')
            if len(outputs) == 0:
                continue
            all_active_time_bins = []
            clfier_preds = []
            species = []
            for k, v in outputs.items():
                all_active_time_bins.append(v['time_bins_exceeding_threshold'])
                clfier_preds.append(v['classifier_predictions'])
                species.append(k)
                
            width = max(max(a) for a in all_active_time_bins) + 1 
            clfier_preds_np = np.zeros([len(clfier_preds), width])
            for i, (j,pred) in enumerate(zip(all_active_time_bins, clfier_preds)):
                clfier_preds_np[i][j] = pred
            
            active_time_bins = np.where(np.max(clfier_preds_np, axis=0))[0]
            active_species = np.array(species)[
                np.argmax(clfier_preds_np, axis=0)[active_time_bins]
                ].tolist()
            
            df_dict['start'].extend(
                (active_time_bins * seg_len).tolist()
            )
            df_dict['end'].extend(
                ((active_time_bins * seg_len) + seg_len).tolist()
            )
            df_dict['audiofilename'].extend(
                [
                    relative_audio_stems[corresponding_audio_file_bool][0] 
                    + '.' 
                    +relative_audio_suffixes[corresponding_audio_file_bool][0]
                    ] * len(active_species)
            )
            df_dict['label:default_classifier'].extend(active_species)
        self.cumulative_annotations = pd.DataFrame(df_dict)
        self.cumulative_annotations = pd.concat(
                [self.cumulative_annotations, clfier_annotations], 
                ignore_index=True
            )
        
    def save_annotation_table(self, loader_obj):
        save_path = (
            self.paths.class_path 
            / f"{loader_obj.model_name}_classifier_annotations.csv"
            )
        self.cumulative_annotations.to_csv(save_path, index=False)
        
    def save_classifier_outputs(self, fileloader_obj, file):
        relative_parent_path = Path(file).relative_to(fileloader_obj.audio_dir).parent
        results_path = self.paths.class_path.joinpath(
            "original_classifier_outputs"
        ).joinpath(relative_parent_path)
        results_path.mkdir(exist_ok=True, parents=True)
        file_dest = results_path.joinpath(file.stem + "_" + self.model_name)
        file_dest = str(file_dest) + ".json"

        if self.predictions.shape[0] != len(self.model.classes):
            self.predictions = self.predictions.swapaxes(0, 1)

        if self.model.only_embed_annotations: #annotation file exists
            np.save(file_dest.replace('.json', '.npy'), self.predictions)
        
        self._fill_dataframe_with_classiefier_results(fileloader_obj, file)
        
        cls_results = self.make_classification_dict(
            self.predictions, self.model.classes, self.classifier_threshold
        )

        with open(file_dest, "w") as f:
            json.dump(cls_results, f, indent=2)
        self.predictions = torch.tensor([])
        
    def run_default_classifier(self, loader):
        all_embeds = loader.embeddings()
        
        for f_name, embeddings in tqdm(
            all_embeds.items(),
            desc='Running pretrained classifier',
            total=len(all_embeds)
            ):        
            
            clfier_output = self.model.classifier_predictions(embeddings)
            if isinstance(clfier_output, torch.Tensor):
                self.predictions = torch.cat(
                    [self.predictions, clfier_output.clone().detach()]
                )
            else:
                self.predictions = torch.cat(
                    [self.predictions, torch.Tensor(clfier_output)]
                )

            self.save_classifier_outputs(loader, loader.audio_dir / f_name)
        
        if loader.model_name in bacpipe.TF_MODELS:
            import tensorflow as tf
            tf.keras.backend.clear_session()