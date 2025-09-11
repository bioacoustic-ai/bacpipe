import librosa as lb
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import time
import torch
import logging
import importlib
import json

logger = logging.getLogger("bacpipe")


class Loader:
    def __init__(
        self,
        audio_dir,
        check_if_combination_exists=True,
        model_name=None,
        dim_reduction_model=False,
        testing=False,
        **kwargs,
    ):
        """
                Run the embedding generation pipeline, check if embeddings for this
        dataset have already been processed, if so load them, if not generate them. 
        During this process collect metadata and return a dictionary of model-
        specific loader objects that can be used to access the embeddings 
        and view metadata. 

        Parameters
        ----------
        audio_dir : string or pathlib.Path
            path to audio data
        check_if_combination_exists : bool, optional
            If false new embeddings are created and the checking is skipped, by default True
        model_name : string, optional
            Name of the model that should be used, by default None
        dim_reduction_model : bool, optional
            Either false if primary embeddings are created or the name of
            the dimensionaliry reduction model if dim reduction should be 
            performed, by default False
        testing : bool, optional
            Testing yes or no?, by default False
        """
        self.model_name = model_name
        self.audio_dir = Path(audio_dir)
        self.dim_reduction_model = dim_reduction_model
        self.testing = testing

        self.initialize_path_structure(testing=testing, **kwargs)

        self.check_if_combination_exists = check_if_combination_exists
        if self.dim_reduction_model:
            self.embed_suffix = ".json"
        else:
            self.embed_suffix = ".npy"
        
        start = time.time()
        self.check_embeds_already_exist()
        logger.debug(
            f"Checking if embeddings already exist took {time.time()-start:.2f}s."
        )

        if self.combination_already_exists or self.dim_reduction_model:
            self.get_embeddings()
        else:
            self._get_audio_paths()
            self._init_metadata_dict()

        if not self.combination_already_exists:
            self.embed_dir.mkdir(exist_ok=True, parents=True)
        else:
            logger.debug(
                "Combination of {} and {} already "
                "exists -> using saved embeddings in {}".format(
                    self.model_name, Path(self.audio_dir).stem, str(self.embed_dir)
                )
            )

    def initialize_path_structure(self, testing=False, **kwargs):
        if testing:
            kwargs["main_results_dir"] = "bacpipe/tests/results_files"

        for key, val in kwargs.items():
            if key == "main_results_dir":
                continue
            if key in ["embed_parent_dir", "dim_reduc_parent_dir", "evaluations_dir"]:
                val = (
                    Path(kwargs["main_results_dir"])
                    .joinpath(self.audio_dir.stem)
                    .joinpath(val)
                )
                val.mkdir(exist_ok=True, parents=True)
            setattr(self, key, val)

    def check_embeds_already_exist(self):
        self.combination_already_exists = False
        self.dim_reduc_embed_dir = False

        if self.check_if_combination_exists:
            if self.dim_reduction_model:
                existing_embed_dirs = Path(self.dim_reduc_parent_dir).iterdir()
            else:
                existing_embed_dirs = Path(self.embed_parent_dir).iterdir()
            if self.testing:
                return
            existing_embed_dirs = list(existing_embed_dirs)
            if isinstance(self.check_if_combination_exists, str):
                existing_embed_dirs = [
                    existing_embed_dirs[0].parent.joinpath(
                        self.check_if_combination_exists
                    )
                ]
            existing_embed_dirs.sort()
            self._find_existing_embed_dir(existing_embed_dirs)

    def _find_existing_embed_dir(self, existing_embed_dirs):
        for d in existing_embed_dirs[::-1]:

            if self.model_name in d.stem and Path(self.audio_dir).stem in d.parts[-1]:
                if list(d.glob("*yml")) == []:
                    try:
                        d.rmdir()
                        continue
                    except OSError:
                        logger.info(
                            f"Directory {d} is not empty. ",
                            "Please remove it manually.",
                        )
                        continue
                with open(d.joinpath("metadata.yml"), "r") as f:
                    mdata = yaml.load(f, Loader=yaml.CLoader)
                    if not self.model_name == mdata["model_name"]:
                        continue

                if self.dim_reduction_model:
                    if self.dim_reduction_model in d.stem:
                        self.combination_already_exists = True
                        logger.info(
                            "\n### Embeddings already exist. "
                            f"Using embeddings in {str(d)} ###"
                        )
                        self.embed_dir = d
                        break
                    else:
                        return d
                else:
                    try:
                        num_files = len(
                            [f for f in list(d.rglob(f"*{self.embed_suffix}"))]
                        )
                        num_audio_files = len(self._get_audio_files())
                    except AssertionError as e:
                        self._get_metadata_dict(d)
                        self.combination_already_exists = True
                        logger.info(
                            f"Error: {e}. "
                            "Will proceed without veryfying if the number of embeddings "
                            "is the same as the number of audio files."
                        )
                        logger.info(
                            "\n### Embeddings already exist. "
                            f"Using embeddings in {self.metadata_dict['embed_dir']} ###"
                        )
                        break

                    if num_audio_files == num_files:
                        self.combination_already_exists = True
                        self._get_metadata_dict(d)
                        logger.info(
                            "\n### Embeddings already exist. "
                            f"Using embeddings in {self.metadata_dict['embed_dir']} ###"
                        )
                        break
                    elif (
                        np.round(num_files / num_audio_files, 1) == 1
                        and num_files > 100
                    ):
                        self.combination_already_exists = True
                        self._get_metadata_dict(d)
                        logger.info(
                            "\n### Embeddings already exist. "
                            f"The number of audio files ({num_audio_files}) "
                            f"and the number of embeddings files ({num_files}) don't "
                            "exactly match. That could be down to some of the audio files "
                            "being corrupt. If you changed the source files and want the "
                            f"embeddings to be computed again, delete or move {d.stem}. \n\n"
                            f"Using embeddings in {self.metadata_dict['embed_dir']} ###"
                        )
                        break

    def _get_audio_paths(self):
        self.files = self._get_audio_files()
        self.files.sort()
        if False:
            self._get_annotation_files()

        self.embed_dir = Path(self.embed_parent_dir).joinpath(self.get_timestamp_dir())

    def _get_annotation_files(self):
        all_annotation_files = list(self.audio_dir.rglob("*.csv"))
        audio_stems = [file.stem for file in self.files]
        self.annot_files = [
            file for file in all_annotation_files if file.stem in audio_stems
        ]

    def _get_audio_files(self):
        if self.audio_dir == 'bacpipe/tests/test_data':
            import importlib.resources as pkg_resources
            with pkg_resources.path(__package__ + ".test_data", "") as audio_dir:
                audio_dir = Path(audio_dir)
        files_list = []
        [
            [files_list.append(ll) for ll in self.audio_dir.rglob(f"*{string}")]
            for string in self.audio_suffixes
        ]
        files_list = np.unique(files_list).tolist()
        assert len(files_list) > 0, "No audio files found in audio_dir."
        return files_list

    def _init_metadata_dict(self):
        self.metadata_dict = {
            "model_name": self.model_name,
            "audio_dir": str(self.audio_dir),
            "embed_dir": str(self.embed_dir),
            "files": {
                "audio_files": [],
                "file_lengths (s)": [],
                "nr_embeds_per_file": [],
            },
        }

    def _get_metadata_dict(self, folder):
        with open(folder.joinpath("metadata.yml"), "r") as f:
            self.metadata_dict = yaml.load(f, Loader=yaml.CLoader)
        for key, val in self.metadata_dict.items():
            if isinstance(val, str):
                if not Path(val).is_dir():
                    if key == "embed_dir":
                        val = folder.parent.joinpath(Path(val).stem)
                    elif key == "audio_dir":
                        logger.info(
                            "The audio files are no longer where they used to be "
                            "during the previous run. This might cause a problem."
                        )
                setattr(self, key, Path(val))
        if self.dim_reduction_model:
            self.dim_reduc_embed_dir = folder

    def get_embeddings(self):
        embed_dir = self.get_embedding_dir()
        self.files = [f for f in embed_dir.rglob(f"*{self.embed_suffix}")]
        self.files.sort()

        if not self.combination_already_exists:
            self._get_metadata_dict(embed_dir)
            self.metadata_dict["files"].update(
                {"embedding_files": [], "embedding_dimensions": []}
            )
            self.embed_dir = Path(self.dim_reduc_parent_dir).joinpath(
                self.get_timestamp_dir() + f"-{self.model_name}"
            )
        else:
            self.embed_dir = embed_dir

    def get_embedding_dir(self):
        if self.dim_reduction_model:
            if self.combination_already_exists:
                self.embed_parent_dir = Path(self.dim_reduc_parent_dir)
                return self.embed_dir
            else:
                self.embed_parent_dir = Path(self.embed_parent_dir)
                self.embed_suffix = ".npy"
        else:
            return self.embed_dir
        self.audio_dir = Path(self.audio_dir)

        if self.dim_reduc_embed_dir:
            # check if they are compatible
            return self.dim_reduc_embed_dir

        embed_dirs = [
            d
            for d in self.embed_parent_dir.iterdir()
            if self.audio_dir.stem in d.parts[-1] and self.model_name in d.stem
        ]
        # check if timestamp of umap is after timestamp of model embeddings
        embed_dirs.sort()
        return self._find_existing_embed_dir(embed_dirs)

    def get_timestamp_dir(self):
        if self.dim_reduction_model:
            model_name = self.dim_reduction_model
        else:
            model_name = self.model_name
        return time.strftime(
            "%Y-%m-%d_%H-%M___" + model_name + "-" + self.audio_dir.stem,
            time.localtime(),
        )

    def embed_read(self, index, file):
        embeds = np.load(file)
        try:
            rel_file_path = file.relative_to(self.metadata_dict["embed_dir"])
        except ValueError as e:
            logger.debug(
                "\nEmbedding file is not in the same directory structure "
                "as it was when created.\n",
                e,
            )
            rel_file_path = file.relative_to(
                self.embed_parent_dir.joinpath(
                    Path(self.metadata_dict["embed_dir"]).stem
                )
            )
        self.metadata_dict["files"]["embedding_files"].append(str(rel_file_path))
        if len(embeds.shape) == 1:
            embeds = np.expand_dims(embeds, axis=0)
        self.metadata_dict["files"]["embedding_dimensions"].append(embeds.shape)
        return embeds

    def embedding_dict(self):
        d = {}
        for file in self.files:
            if not self.dim_reduction_model:
                embeds = np.load(file)
            else:
                with open(file, "r") as f:
                    embeds = json.load(f)
                embeds = np.array(embeds)
            d[str(file.relative_to(self.embed_dir))] = embeds
        return d

    def write_audio_file_to_metadata(self, index, file, embed, embeddings):
        if (
            not "segment_length (samples)" in self.metadata_dict.keys()
            or not "sample_rate (Hz)" in self.metadata_dict.keys()
            or not "embedding_size" in self.metadata_dict.keys()
        ):
            self.metadata_dict["segment_length (samples)"] = embed.model.segment_length
            self.metadata_dict["sample_rate (Hz)"] = embed.model.sr
            self.metadata_dict["embedding_size"] = embeddings.shape[-1]
        rel_file_path = Path(file).relative_to(self.audio_dir)
        self.metadata_dict["files"]["audio_files"].append(str(rel_file_path))
        self.metadata_dict["files"]["file_lengths (s)"].append(
            embed.file_length[file.stem]
        )
        self.metadata_dict["files"]["nr_embeds_per_file"].append(embeddings.shape[0])

    def write_metadata_file(self):
        self.metadata_dict["nr_embeds_total"] = sum(
            self.metadata_dict["files"]["nr_embeds_per_file"]
        )
        self.metadata_dict["total_dataset_length (s)"] = sum(
            self.metadata_dict["files"]["file_lengths (s)"]
        )
        with open(str(self.embed_dir.joinpath("metadata.yml")), "w") as f:
            yaml.safe_dump(self.metadata_dict, f)

    def update_files(self):
        if self.dim_reduction_model:
            self.files = [f for f in self.embed_dir.iterdir() if f.suffix == ".json"]
        else:
            self.files = list(self.embed_dir.rglob("*.npy"))


import queue
import threading
from tqdm import tqdm


class Embedder:
    def __init__(
        self,
        model_name,
        dim_reduction_model=False,
        paths=None,
        classifier_threshold=None,
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
        dim_reduction_model : bool, optional
            Can be bool or the string corresponding to the dimensionality reduction model, by default False
        paths : SimpleNamespace, optional
            dict-like structure with all the results paths, by default None
        testing : bool, optional
            _description_, by default False
        classifier_threshold : float, optional
            Value under which class predictions are discarded, by default None
        """
        self.paths = paths
        self.file_length = {}

        if classifier_threshold:
            self.classifier_threshold = classifier_threshold

        self.dim_reduction_model = dim_reduction_model
        if dim_reduction_model:
            self.dim_reduction_model = True
            self.model_name = dim_reduction_model
        else:
            self.model_name = model_name
        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        """
        Load model specific module, instantiate model and allocate device for model.
        """
        if self.dim_reduction_model:
            module = importlib.import_module(
                f"bacpipe.embedding_generation_pipelines.dimensionality_reduction.{self.model_name}"
            )
        else:
            module = importlib.import_module(
                f"bacpipe.embedding_generation_pipelines.feature_extractors.{self.model_name}"
            )
        self.model = module.Model(**kwargs)
        self.model.prepare_inference()

    def prepare_audio(self, sample):
        """
        Use bacpipe pipeline to load audio file, window it according to 
        model specific window length and preprocess the data, ready for 
        batch inference computation. Also log file length and shape for
        metadata files.

        Parameters
        ----------
        sample : pathlib.Path or str
            path to audio file

        Returns
        -------
        torch.Tensor
            audio frames preprocessed with model specific preprocessing
        """
        audio = self.model.load_and_resample(sample)
        frames = self.model.window_audio(audio)
        preprocessed_frames = self.model.preprocess(frames)
        self.file_length[sample.stem] = len(audio[0]) / self.model.sr
        self.preprocessed_shape = tuple(preprocessed_frames.shape)
        return preprocessed_frames

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
        batched_samples = self.model.init_dataloader(sample)
        embeds = self.model.batch_inference(batched_samples)
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

    def get_pipelined_embeddings_from_model(self, fileloader_obj):
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
            for idx, file in enumerate(fileloader_obj.files):
                try:
                    preprocessed = self.prepare_audio(file)
                    task_queue.put((idx, file, preprocessed))
                except Exception as e:
                    task_queue.put((idx, file, e))
            task_queue.put(None)  # sentinel = done

        threading.Thread(target=producer, daemon=True).start()

        # --- Consumer: embed + save metadata/embeddings ---
        with tqdm(
            total=len(fileloader_obj.files),
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
                except Exception as e:
                    logger.warning(
                        f"Error generating embeddings for {file}, skipping file.\nError: {e}"
                    )
                    pbar.update(1)
                    continue

                fileloader_obj.write_audio_file_to_metadata(idx, file, self, embeddings)
                self.save_embeddings(idx, fileloader_obj, file, embeddings)
                if self.model.bool_classifier:
                    self.save_classifier_outputs(fileloader_obj, file)

                pbar.update(1)

        return fileloader_obj  # same return type as sequential version

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
            sample = self.prepare_audio(sample)
            embeds = self.get_embeddings_for_audio(sample)

        logger.debug(f"{self.model_name} embeddings have shape: {embeds.shape}")
        logger.info(f"{self.model_name} inference took {time.time()-start:.2f}s.")
        return embeds

    def save_embeddings(self, file_idx, fileloader_obj, file, embeds):
        if self.dim_reduction_model:
            file_dest = fileloader_obj.embed_dir.joinpath(
                fileloader_obj.audio_dir.stem + "_" + self.model_name
            )
            file_dest = str(file_dest) + ".json"
            input_len = (
                fileloader_obj.metadata_dict["segment_length (samples)"]
                / fileloader_obj.metadata_dict["sample_rate (Hz)"]
            )
            save_embeddings_dict_with_timestamps(
                file_dest, embeds, input_len, fileloader_obj, file_idx
            )
        else:
            relative_parent_path = (
                Path(file).relative_to(fileloader_obj.audio_dir).parent
            )
            parent_path = fileloader_obj.embed_dir.joinpath(relative_parent_path)
            parent_path.mkdir(exist_ok=True, parents=True)
            file_dest = parent_path.joinpath(file.stem + "_" + self.model_name)
            file_dest = str(file_dest) + ".npy"
            if len(embeds.shape) == 1:
                embeds = np.expand_dims(embeds, axis=0)
            np.save(file_dest, embeds)

    @staticmethod
    def make_classification_dict(probabilities, classes, threshold):
        if probabilities.shape[0] != len(classes):
            probabilities = probabilities.swapaxes(0, 1)

        cls_idx, tmp_idx = np.where(probabilities > threshold)

        cls_results = {
            classes[k]: {
                "time_bins_exceeding_threshold": tmp_idx[cls_idx == k].tolist(),
                "classifier_predictions": np.array(
                    probabilities[cls_idx[cls_idx == k], tmp_idx[cls_idx == k]]
                ).tolist(),
            }
            for k in cls_idx
        }

        cls_results["head"] = {
            "Time bins in this file": probabilities.shape[0],
            "Threshold for classifier predictions": threshold,
        }
        return cls_results

    def save_classifier_outputs(self, fileloader_obj, file):
        relative_parent_path = Path(file).relative_to(fileloader_obj.audio_dir).parent
        results_path = self.paths.class_path.joinpath(
            "original_classifier_outputs"
        ).joinpath(relative_parent_path)
        results_path.mkdir(exist_ok=True, parents=True)
        file_dest = results_path.joinpath(file.stem + "_" + self.model_name)
        file_dest = str(file_dest) + ".json"

        if self.model.classifier_outputs.shape[0] != len(self.model.classes):
            self.model.classifier_outputs = self.model.classifier_outputs.swapaxes(0, 1)

        cls_results = self.make_classification_dict(
            self.model.classifier_outputs, self.model.classes, self.classifier_threshold
        )

        classifier_annotations = pd.DataFrame()

        tmp_bins = self.model.classifier_outputs.shape[-1]
        classifier_annotations["start"] = np.arange(tmp_bins) * (
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
        )[torch.argmax(self.model.classifier_outputs, dim=0)].tolist()

        if not hasattr(self, "cumulative_annotations"):
            self.cumulative_annotations = classifier_annotations
        else:
            self.cumulative_annotations = pd.concat(
                [self.cumulative_annotations, classifier_annotations], ignore_index=True
            )

        with open(file_dest, "w") as f:
            json.dump(cls_results, f, indent=2)
        self.model.classifier_outputs = torch.tensor([])


def save_embeddings_dict_with_timestamps(
    file_dest, embeds, input_len, loader_obj, f_idx
):

    t_stamps = []
    for num_segments, _ in loader_obj.metadata_dict["files"]["embedding_dimensions"]:
        [t_stamps.append(t) for t in np.arange(0, num_segments * input_len, input_len)]
    d = {
        var: embeds[:, i].tolist() for i, var in zip(range(embeds.shape[1]), ["x", "y"])
    }
    d["timestamp"] = t_stamps

    d["metadata"] = {
        k: (v if isinstance(v, list) else v)
        for (k, v) in loader_obj.metadata_dict["files"].items()
    }
    d["metadata"].update(
        {k: v for (k, v) in loader_obj.metadata_dict.items() if not isinstance(v, dict)}
    )

    import json

    with open(file_dest, "w") as f:
        json.dump(d, f, indent=2)

    if embeds.shape[-1] > 2:
        embed_dict = {}
        acc_shape = 0
        for shape, file in zip(
            loader_obj.metadata_dict["files"]["embedding_dimensions"],
            loader_obj.files,
        ):
            embed_dict[file.stem] = embeds[acc_shape : acc_shape + shape[0]]
            acc_shape += shape[0]
        np.save(file_dest.replace(".json", f"{embeds.shape[-1]}.npy"), embed_dict)
