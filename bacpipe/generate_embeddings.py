import librosa as lb
import numpy as np
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import logging
import importlib

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
        self.model_name = model_name
        self.audio_dir = audio_dir
        self.dim_reduction_model = dim_reduction_model

        with open("bacpipe/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        for key, val in self.config.items():
            setattr(self, key, val)

        self.check_if_combination_exists = check_if_combination_exists
        if self.dim_reduction_model:
            self.embed_suffix = ".json"
        else:
            self.embed_suffix = ".npy"

        self.check_embeds_already_exist()
        if self.combination_already_exists or self.dim_reduction_model:
            self.get_embeddings()
        else:
            self._get_audio_paths()
            self._init_metadata_dict()

        if not self.combination_already_exists and not testing:
            self.embed_dir.mkdir(exist_ok=True, parents=True)
        else:
            logger.debug(
                "Combination of {} and {} already "
                "exists -> using saved embeddings in {}".format(
                    self.model_name, Path(self.audio_dir).stem, str(self.embed_dir)
                )
            )

    def check_embeds_already_exist(self):
        self.combination_already_exists = False
        self.dim_reduc_embed_dir = False

        if self.check_if_combination_exists:
            if self.dim_reduction_model:
                existing_embed_dirs = list(Path(self.dim_reduc_parent_dir).iterdir())
            else:
                existing_embed_dirs = list(Path(self.embed_parent_dir).iterdir())
            if isinstance(self.check_if_combination_exists, str):
                existing_embed_dirs = [
                    existing_embed_dirs[0].parent.joinpath(
                        self.check_if_combination_exists
                    )
                ]
            existing_embed_dirs.sort()
            for d in existing_embed_dirs[::-1]:

                if (
                    self.model_name in d.stem
                    and Path(self.audio_dir).stem in d.stem
                    and self.model_name in d.stem
                ):

                    num_files = len(
                        [f for f in d.iterdir() if f.suffix == self.embed_suffix]
                    )
                    num_audio_files = len(
                        [
                            f
                            for f in Path(self.audio_dir).iterdir()
                            if f.suffix in self.config["audio_suffixes"]
                        ]
                    )

                    if num_audio_files == num_files:
                        self.combination_already_exists = True
                        self._get_metadata_dict(d)
                        break

    def _get_audio_paths(self):
        self.audio_dir = Path(self.audio_dir)

        self.files = self._get_audio_files()

        self.embed_dir = Path(self.embed_parent_dir).joinpath(self.get_timestamp_dir())

    def _get_audio_files(self):
        files_list = []
        [
            [files_list.append(ll) for ll in self.audio_dir.rglob(f"*{string}")]
            for string in self.config["audio_suffixes"]
        ]
        return files_list

    def _init_metadata_dict(self):
        self.metadata_dict = {
            "model_name": self.model_name,
            "audio_dir": str(self.audio_dir),
            "embed_dir": str(self.embed_dir),
            "files": {"audio_files": [], "file_lengths (s)": [], "preproc_shape": []},
        }

    def _get_metadata_dict(self, folder):
        with open(folder.joinpath("metadata.yml"), "r") as f:
            self.metadata_dict = yaml.safe_load(f)
        for key, val in self.metadata_dict.items():
            if isinstance(val, str) and Path(val).is_dir():
                setattr(self, key, Path(val))
        if self.model_name == "umap":
            self.dim_reduc_embed_dir = folder

    def get_embeddings(self):
        embed_dir = self.get_embedding_dir()
        self.files = [f for f in embed_dir.iterdir() if f.suffix == self.embed_suffix]
        self.files.sort()
        if not self.combination_already_exists:
            self._get_metadata_dict(embed_dir)
            self.metadata_dict["files"].update(
                {"embedding_files": [], "embedding_dimensions": []}
            )
            self.embed_dir = Path(self.dim_reduc_parent_dir).joinpath(
                self.get_timestamp_dir() + f"-{self.model_name}"
            )

    def get_embedding_dir(self):
        if self.dim_reduction_model:
            if self.combination_already_exists:
                self.embed_parent_dir = Path(self.dim_reduc_parent_dir)
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
            if self.audio_dir.stem in d.stem and self.model_name in d.stem
        ]
        # check if timestamp of umap is after timestamp of model embeddings
        embed_dirs.sort()
        most_recent_emdbed_dir = embed_dirs[-1]
        return most_recent_emdbed_dir

    def get_annotations(self):
        pass

    def get_timestamp_dir(self):
        if self.dim_reduction_model:
            model_name = self.dim_reduction_model
        else:
            model_name = self.model_name
        return time.strftime(
            "%Y-%m-%d_%H-%M___" + model_name + "-" + self.audio_dir.stem,
            time.localtime(),
        )

    def embed_read(self, file):
        embeds = np.load(file)
        self.metadata_dict["files"]["embedding_files"].append(str(file))
        self.metadata_dict["files"]["embedding_dimensions"].append(str(embeds.shape))
        return embeds

    def write_audio_file_to_metadata(self, file, embed):
        if not self.dim_reduction_model:
            self.metadata_dict["segment_length (samples)"] = embed.model.segment_length
            self.metadata_dict["sample_rate (Hz)"] = embed.model.sr
            self.metadata_dict["files"]["audio_files"].append(file.stem + file.suffix)
            self.metadata_dict["files"]["file_lengths (s)"].append(embed.file_length)
            self.metadata_dict["files"]["preproc_shape"].append(
                str(embed.preprocessed_shape)
            )

    def write_metadata_file(self):
        with open(str(self.embed_dir.joinpath("metadata.yml")), "w") as f:
            yaml.safe_dump(self.metadata_dict, f)

    def update_files(self):
        if self.dim_reduction_model:
            self.files = [f for f in self.embed_dir.iterdir() if f.suffix == ".json"]


class Embedder:
    def __init__(self, model_name, dim_reduction_model=False, **kwargs):
        import yaml

        with open("bacpipe/config.yaml", "rb") as f:
            self.config = yaml.safe_load(f)

        self.dim_reduction_model = dim_reduction_model
        if dim_reduction_model:
            self.dim_reduction_model = True
            self.model_name = dim_reduction_model
        else:
            self.model_name = model_name
        self._init_model()

    def _init_model(self):
        module = importlib.import_module(f"bacpipe.pipelines.{self.model_name}")
        self.model = module.Model()

    def prepare_audio(self, sample):
        audio = self.model.load_and_resample(sample)
        frames = self.model.window_audio(audio)
        preprocessed_frames = self.model.preprocess(frames)
        self.file_length = len(audio[0]) / self.model.sr
        self.preprocessed_shape = tuple(preprocessed_frames.shape)
        return preprocessed_frames

    def get_embeddings_for_audio(self, sample):
        batched_samples = self.model.init_dataloader(sample)
        embeds = self.model.batch_inference(batched_samples)
        if not isinstance(embeds, np.ndarray):
            try:
                embeds = embeds.numpy()
            except:
                embeds = embeds.detach().numpy()
        return embeds

    def get_reduced_dimensionality_embeddings(self, embeds):
        samples = self.model.preprocess(embeds)
        return self.model(samples)

    def get_embeddings_from_model(self, sample):

        start = time.time()
        if self.dim_reduction_model:
            embeds = self.get_reduced_dimensionality_embeddings(sample)
        else:
            sample = self.prepare_audio(sample)
            embeds = self.get_embeddings_for_audio(sample)

        logger.debug(f"{self.model_name} embeddings have shape: {embeds.shape}")
        logger.info(f"{self.model_name} inference took {time.time()-start:.2f}s.")
        return embeds

    def save_embeddings(self, file_idx, fileloader_obj, file, embeds):
        file_dest = fileloader_obj.embed_dir.joinpath(file.stem + "_" + self.model_name)
        if file.suffix == ".npy":
            file_dest = str(file_dest) + ".json"
            input_len = (
                fileloader_obj.metadata_dict["segment_length (samples)"]
                / fileloader_obj.metadata_dict["sample_rate (Hz)"]
            )
            save_embeddings_dict_with_timestamps(
                file_dest, embeds, input_len, fileloader_obj, file_idx
            )
            # TODO save png of embeddings for umap embeds
        else:
            file_dest = str(file_dest) + ".npy"
            np.save(file_dest, embeds)


def save_embeddings_dict_with_timestamps(
    file_dest, embeds, input_len, loader_obj, f_idx
):
    length = embeds.shape[0]
    lin_array = np.arange(0, length * input_len, input_len)
    d = {
        var: embeds[:, i].tolist() for i, var in zip(range(embeds.shape[1]), ["x", "y"])
    }
    d["timestamp"] = lin_array.tolist()

    d["metadata"] = {
        k: (v[f_idx] if isinstance(v, list) else v)
        for (k, v) in loader_obj.metadata_dict["files"].items()
    }
    d["metadata"].update(
        {k: v for (k, v) in loader_obj.metadata_dict.items() if not isinstance(v, dict)}
    )

    import json

    with open(file_dest, "w") as f:
        json.dump(d, f)


def generate_embeddings(save_files=True, **kwargs):
    ld = Loader(**kwargs)
    if not ld.combination_already_exists:
        embed = Embedder(**kwargs)
        for idx, file in tqdm(enumerate(ld.files)):
            if file.suffix == ".npy":
                sample = ld.embed_read(file)
            else:
                sample = file
            embeddings = embed.get_embeddings_from_model(sample)
            ld.write_audio_file_to_metadata(file, embed)
            embed.save_embeddings(idx, ld, file, embeddings)
        ld.write_metadata_file()
        ld.update_files()
    return ld
