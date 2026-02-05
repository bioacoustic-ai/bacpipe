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
import os
import queue
import threading
from tqdm import tqdm
import torchaudio as ta
import bacpipe
import soundfile

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
        self.continue_failed_run = False

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

        if self.continue_failed_run:
            self._get_metadata_from_created_embeddings()
        elif not self.combination_already_exists:
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

    def _get_metadata_from_created_embeddings(self):
        module = importlib.import_module(
                f"bacpipe.model_pipelines.feature_extractors.{self.model_name}"
            )
        already_processed_files = list(Path(self.embed_dir).rglob('*.npy'))
        already_processed_files.sort()
        relative_audio_stems = np.array([str(f.relative_to(self.audio_dir)).split('.')[0] for f in self.files])
        audio_files = np.array(self.files)
        audio_suffixes = np.array([f.suffix for f in self.files])
        for file in already_processed_files:
            with open(file, 'rb') as f:
                corresponding_audio_file_bool = (
                    relative_audio_stems==str(
                        file.relative_to(self.embed_dir)
                        ).replace(f'_{self.model_name}.npy', '')
                )
                embed = np.load(f)
                self.metadata_dict['files']['audio_files'].append(
                    relative_audio_stems[corresponding_audio_file_bool][0]
                    + audio_suffixes[corresponding_audio_file_bool][0]
                )
                self.metadata_dict['files']['nr_embeds_per_file'].append(
                    embed.shape[0]
                )
                self.metadata_dict['files']['file_lengths (s)'].append(
                    embed.shape[0] * (module.LENGTH_IN_SAMPLES / module.SAMPLE_RATE)
                )
                self._update_audio_file_list(audio_files, corresponding_audio_file_bool)        
        
    def _update_audio_file_list(self, audio_files, corresponding_audio_file_bool):
        self.files.remove(
            audio_files[corresponding_audio_file_bool][0]
        )

    def _find_existing_embed_dir(self, existing_embed_dirs):
        for d in existing_embed_dirs[::-1]:

            if self.model_name in d.stem and Path(self.audio_dir).stem in d.parts[-1]:
                if list(d.glob("*yml")) == []:
                    try:
                        d.rmdir()
                        continue
                    except OSError:
                        logger.info(
                            f"\nThe directory {d} is not empty. "
                            "It seems like a previous run failed, "
                            "bacpipe is comparing what files were already "
                            "created and will then continue where it left off."
                            "If you interrupted the run on purpose and want to "
                            "start from the beginning, please cancel using "
                            "Ctrl + C and then remove "
                            f"the folder {d} manually.\n"
                        )
                        self.continue_failed_run = True
                        self.embed_dir = d
                        return d
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
                            f"\nError: {e}. "
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
                        np.round(num_files / num_audio_files, 1) == 1 # allow 5 % deviation
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
        if not self.continue_failed_run:
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
                if key == 'model_name':
                    continue
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

    def embeddings(self, as_type='dict'):
        d = {}
        for file in self.files:
            if not self.dim_reduction_model:
                embeds = np.load(file)
            else:
                with open(file, "r") as f:
                    embeds = json.load(f)
                embeds = np.array(embeds)
            d[str(file.relative_to(self.embed_dir))] = embeds
        if as_type == 'dict':
            return d
        elif as_type == 'array':
            return np.array(d.values())

    def write_audio_file_to_metadata(self, file, model, embeddings, file_length):
        if (
            not "segment_length (samples)" in self.metadata_dict.keys()
            or not "sample_rate (Hz)" in self.metadata_dict.keys()
            or not "embedding_size" in self.metadata_dict.keys()
        ):
            self.metadata_dict["segment_length (samples)"] = model.segment_length
            self.metadata_dict["sample_rate (Hz)"] = model.sr
            self.metadata_dict["embedding_size"] = embeddings.shape[-1]
        rel_file_path = Path(file).relative_to(self.audio_dir)
        self.metadata_dict["files"]["audio_files"].append(str(rel_file_path))
        self.metadata_dict["files"]["file_lengths (s)"].append(
            file_length[file.stem]
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
            
    def save_embeddings(self, file_idx, file, embeds):
        if self.dim_reduction_model:
            file_dest = self.embed_dir.joinpath(
                self.audio_dir.stem + "_" + self.model_name
            )
            file_dest = str(file_dest) + ".json"
            input_len = (
                self.metadata_dict["segment_length (samples)"]
                / self.metadata_dict["sample_rate (Hz)"]
            )
            self.save_embeddings_dict_with_timestamps(
                file_dest, embeds, input_len
            )
        else:
            relative_parent_path = (
                Path(file).relative_to(self.audio_dir).parent
            )
            parent_path = self.embed_dir.joinpath(relative_parent_path)
            parent_path.mkdir(exist_ok=True, parents=True)
            file_dest = parent_path.joinpath(file.stem + "_" + self.model_name)
            file_dest = str(file_dest) + ".npy"
            if len(embeds.shape) == 1:
                embeds = np.expand_dims(embeds, axis=0)
            np.save(file_dest, embeds)

    def save_embeddings_dict_with_timestamps(
        self, file_dest, embeds, input_len
    ):
        t_stamps = []
        for num_segments, _ in self.metadata_dict["files"]["embedding_dimensions"]:
            [t_stamps.append(t) for t in np.arange(0, num_segments * input_len, input_len)]
        d = {
            var: embeds[:, i].tolist() for i, var in zip(range(embeds.shape[1]), ["x", "y"])
        }
        d["timestamp"] = t_stamps

        d["metadata"] = {
            k: (v if isinstance(v, list) else v)
            for (k, v) in self.metadata_dict["files"].items()
        }
        d["metadata"].update(
            {k: v for (k, v) in self.metadata_dict.items() if not isinstance(v, dict)}
        )

        import json

        with open(file_dest, "w") as f:
            json.dump(d, f, indent=2)

        if embeds.shape[-1] > 2:
            embed_dict = {}
            acc_shape = 0
            for shape, file in zip(
                self.metadata_dict["files"]["embedding_dimensions"],
                self.files,
            ):
                embed_dict[file.stem] = embeds[acc_shape : acc_shape + shape[0]]
                acc_shape += shape[0]
            np.save(file_dest.replace(".json", f"{embeds.shape[-1]}.npy"), embed_dict)

class AudioHelper:
    def __init__(self, model, padding, audio_dir, **kwargs):
        """
        Helper class for all methods related to loading and padding audio. 

        Parameters
        ----------
        model : Model object
            has attributes for all the model characteristics like 
            sample rate, segment length etc. as well as the methods
            to run the model
        padding : str
            padding function to use for where padding is necessary
        audio_dir : pathlib.Path object
            path to audio dir
        """
        self.model = model
        self.padding = padding
        self.audio_dir = audio_dir
    
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
        audio = self.load_and_resample(sample)
        audio = audio.to(self.model.device)
        if self.model.only_embed_annotations:
            frames = self.only_load_annotated_segments(sample, audio)
        else:
            frames = self.window_audio(audio)
        preprocessed_frames = self.model.preprocess(frames)
        self.file_length[sample.stem] = len(audio[0]) / self.model.sr
        self.preprocessed_shape = tuple(preprocessed_frames.shape)
        if self.model.device == 'cuda':
            del audio, frames
            torch.cuda.empty_cache()
        return preprocessed_frames
    
    def load_and_resample(self, path):
        try:
            audio, sr = ta.load(str(path), normalize=True)
        except Exception as e:
            logger.exception(
                f"\nError loading audio with torchaudio. "
                f"Skipping {path}."
                f"Error: {e}"
            )
            raise e
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0).unsqueeze(0)
        if len(audio[0]) == 0:
            error = f"Audio file {path} is empty. " f"Skipping {path}."
            logger.exception(error)
            raise ValueError(error)
        re_audio = ta.functional.resample(audio, sr, self.model.sr)
        return re_audio

    def only_load_annotated_segments(self, file_path, audio):
        import pandas as pd
        annots = pd.read_csv(Path(self.audio_dir) / 'annotations.csv')
        # filter current file
        file_annots = annots[annots.audiofilename==file_path.relative_to(self.audio_dir)]
        if len(file_annots) == 0:
            file_annots = annots[annots.audiofilename==file_path.stem+file_path.suffix]
        
        starts = np.array(file_annots.start, dtype=np.float32)*self.model.sr
        ends = np.array(file_annots.end, dtype=np.float32)*self.model.sr

        audio = audio.cpu().squeeze()
        for idx, (s, e) in enumerate(zip(starts, ends)):
            s, e = int(s), int(e)
            if e > len(audio):
                logger.warning(
                    f"Annotation with start {s} and end {e} is outside of "
                    f"range of {file_path}. Skipping annotation."
                )
                continue
            segments = lb.util.fix_length(
                audio[s:e+1],
                size=self.model.segment_length,
                mode=self.padding
                )
            if idx == 0:
                cumulative_segments = segments
            else:
                cumulative_segments = np.vstack([cumulative_segments, segments])
        cumulative_segments = torch.Tensor(cumulative_segments)
        cumulative_segments = cumulative_segments.to(self.device)
        return cumulative_segments

    def window_audio(self, audio):
        num_frames = int(np.ceil(len(audio[0]) / self.model.segment_length))
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu()
        padded_audio = lb.util.fix_length(
            audio,
            size=int(num_frames * self.model.segment_length),
            mode=self.padding,
        )
        logger.debug(f"{self.padding} was used on an audio segment.")
        frames = padded_audio.reshape([num_frames, self.model.segment_length])
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames)
        frames = frames.to(self.model.device)
        return frames

class Embedder(AudioHelper):
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

            return tf.data.Dataset.from_tensor_slices(audio).batch(self.model.batch_size)
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
                    and not isinstance(batch, tensorflow.Tensor)
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

    def get_dimensionality_reduced_embeddings_pipeline(self):
        for idx, file in enumerate(
            tqdm(self.loader.files, desc="processing files", position=1, leave=False)
        ):
            if idx == 0:
                embeddings = self.loader.embed_read(idx, file)
            else:
                embeddings = np.concatenate(
                    [embeddings, self.loader.embed_read(idx, file)]
                )

        dim_reduced_embeddings = self.get_embeddings_from_model(embeddings)
        self.loader.save_embeddings(idx, file, dim_reduced_embeddings)

    def get_embeddings_using_multithreading_pipeline(self):
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
                except Exception as e:
                    logger.warning(
                        f"Error generating embeddings for {file}, skipping file.\nError: {e}"
                    )
                    pbar.update(1)
                    continue
            
                self.loader.write_audio_file_to_metadata(
                    idx, self.model, embeddings, self.file_length
                    )
                self.loader.save_embeddings(idx, file, embeddings)
                if self.model.bool_classifier:
                    self.classifier.save_classifier_outputs(self.loader, file)

                pbar.update(1)

    def get_embeddings_sequentially_pipeline(self):
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
            
            self.loader.write_audio_file_to_metadata(
                file, self.model, embeddings, self.file_length
                )
            self.loader.save_embeddings(idx, file, embeddings)
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
            self.classifier = self.classifier.cuda()

        if isinstance(clfier_output, torch.Tensor):
            self.predictions = torch.cat(
                [self.predictions, clfier_output.clone().detach()]
            )
        else:
            self.predictions = torch.cat(
                [self.predictions, torch.Tensor(clfier_output)]
            )
    
    def run_clfier_for_previous_embeddings(self, fileloader_obj):
        existing_embeddings_files = list(
            Path(fileloader_obj.embed_dir).rglob('*.npy')
            )
        for file in existing_embeddings_files:
            with open(file, 'rb') as f:
                embeddings = np.load(f)
            self.classify(embeddings)
                
        
                
    
    def fill_dataframe_with_classiefier_results(self, fileloader_obj, file):
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
            if fileloader_obj.continue_failed_run:
                self.load_existing_clfier_outputs(fileloader_obj, 
                                                  classifier_annotations)
            else:
                self.cumulative_annotations = classifier_annotations
        else:
            self.cumulative_annotations = pd.concat(
                [self.cumulative_annotations, classifier_annotations], ignore_index=True
            )

    def load_existing_clfier_outputs(self, fileloader_obj, clfier_annotations):
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
        
        self.fill_dataframe_with_classiefier_results(fileloader_obj, file)
        
        cls_results = self.make_classification_dict(
            self.predictions, self.model.classes, self.classifier_threshold
        )

        with open(file_dest, "w") as f:
            json.dump(cls_results, f, indent=2)
        self.predictions = torch.tensor([])