
import json
import yaml
import time
import logging
import importlib
import numpy as np
from pathlib import Path

from bacpipe import settings as bacpipe_settings
logger = logging.getLogger("bacpipe")


def save_logs():
    import datetime
    import json
    from bacpipe import config, settings
    
    log_dir = Path(settings.main_results_dir) / Path(config.audio_dir).stem / f"logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"bacpipe_{timestamp}.log"

    f_format = logging.Formatter(
        "%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s"
    )
    f_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(f_format)
    f_handler.flush = lambda: f_handler.stream.flush()  # optional, for clarity
    logger.addHandler(f_handler)

    # Save current config + settings snapshot
    settings_dict, config_dict = {}, {}
    for k, v in vars(settings).items():
        if '/' in str(v) or '\\' in str(v):
            settings_dict[k] = Path(v).as_posix()
        else:
            settings_dict[k] = v
    for k, v in vars(config).items():
        if '/' in str(v) or '\\' in str(v):
            config_dict[k] = Path(v).as_posix()
        else:
            config_dict[k] = v
    
    with open(
        log_dir / f"config_{timestamp}.json", "w"
    ) as f:
        json.dump(config_dict, f, indent=2)
    with open(
        log_dir / f"settings_{timestamp}.json", "w"
    ) as f:
        json.dump(settings_dict, f, indent=2)

    logger.info("Saved config, settings, and logs to %s", log_dir)

class Loader:
    def __init__(
        self,
        audio_dir,
        model_name=None,
        check_if_combination_exists=True,
        dim_reduction_model=False,
        use_folder_structure=False, 
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
        model_name : string, optional
            Name of the model that should be used, by default None
        check_if_combination_exists : bool, optional
            If false new embeddings are created and the checking is skipped, by default True
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
        self.use_folder_structure = use_folder_structure
        self.continue_incomplete_run = False

        self._initialize_path_structure(
            testing=testing, **kwargs
            )

        self.check_if_combination_exists = check_if_combination_exists

        if self.dim_reduction_model:
            self.embed_suffix = ".json"
        else:
            self.embed_suffix = ".npy"
        
        start = time.time()
        self._check_embeds_already_exist()
        logger.debug(
            f"Checking if embeddings already exist took {time.time()-start:.2f}s."
        )

        if self.combination_already_exists or self.dim_reduction_model:
            self._get_embeddings()
        elif not hasattr(self, 'files'):
            self._get_audio_paths_and_init_embed_dir()
            self._init_metadata_dict()

        if not self.use_folder_structure:
            logger.info(
                "No model_name is passed, therefore no directory "
                "structure will be created."
            )
        elif not self.combination_already_exists:
            self.embed_dir.mkdir(exist_ok=True, parents=True)
        else:
            logger.debug(
                "Combination of {} and {} already "
                "exists -> using saved embeddings in {}".format(
                    self.model_name, Path(self.audio_dir).stem, str(self.embed_dir)
                )
            )

    def _initialize_path_structure(
        self, testing=False, **kwargs
        ):
        # if testing:
        #     kwargs["main_results_dir"] = "bacpipe/tests/results_files"
        
        if self.use_folder_structure:
            from bacpipe import settings
            kwargs = {**vars(settings)}

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

    def _check_embeds_already_exist(self):
        self.combination_already_exists = False
        self.dim_reduc_embed_dir = False

        if not self.check_if_combination_exists:
            return 
        
        if self.dim_reduction_model:
            existing_embed_dirs = Path(self.dim_reduc_parent_dir).iterdir()
        elif not hasattr(self, 'embed_parent_dir'):
            return
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
        relative_audio_stems = np.array([
                str(f.relative_to(self.audio_dir).parent 
                / f.relative_to(self.audio_dir).stem)
             for f in self.files
             ])
        audio_files = np.array(self.files)
        audio_suffixes = np.array([f.suffix for f in self.files])
        for file in already_processed_files:
            with open(file, 'rb') as f:
                corresponding_audio_file_bool = (
                    relative_audio_stems==str(
                        file.relative_to(self.embed_dir)
                        ).replace(f'_{self.model_name}.npy', '')
                )
                try:
                    embed = np.load(f)
                except Exception as e:
                    logger.exception(
                        f"Unable to load file {f}. Continuing with the "
                        f"next file. {e}"
                    )
                    continue
                self.metadata_dict['files']['audio_files'].append(
                    relative_audio_stems[corresponding_audio_file_bool][0]
                    + audio_suffixes[corresponding_audio_file_bool][0]
                )
                self.metadata_dict['files']['nr_embeds_per_file'].append(
                    embed.shape[0]
                )
                self.metadata_dict['files']['file_lengths (s)'].append(
                    embed.shape[0] * (
                        module.LENGTH_IN_SAMPLES / module.SAMPLE_RATE
                        )
                )
                self._update_audio_file_list(
                    audio_files, corresponding_audio_file_bool
                    )
        if len(self.files) == 0:
            self.metadata_dict['segment_length (samples)'] = module.LENGTH_IN_SAMPLES
            self.metadata_dict['sample_rate (Hz)'] = module.SAMPLE_RATE
            self.write_metadata_file()
            self.combination_already_exists = True

        
    def _update_audio_file_list(self, audio_files, corresponding_audio_file_bool):
        self.files.remove(
            audio_files[corresponding_audio_file_bool][0]
        )

    def _find_existing_embed_dir(self, existing_embed_dirs):
        """
        Check if embeddings have already been calculated for this combination
        of model and audio dir. If the combination exists, check if it's empty
        or very incomplete. If empty, delete it. If incomplete continue where
        it was left off. If it exists and contains a metadata.yml file, then
        load the file. The check can be avoided by specifying 
        check_if_combination_exists as False. The function only returns if 
        it is a dimensionality reduction model we are currently working on
        to locate previously computed dimensionality reduced embeddings. 

        Parameters
        ----------
        existing_embed_dirs : list
            list of directories to check if the combination we are trying
            to process is potentially contained in

        Returns
        -------
        pathlib.Path object
            directory containing dimensionality reduced embeddings 
            that were already processed 
        """
        # iterate through directories backwards, starting with most recent first
        for d in existing_embed_dirs[::-1]: 
            # require that the model name and the audio dir are in the folder name
            if not (
                self.model_name in d.stem 
                and not self.combination_already_exists
                and Path(self.audio_dir).stem in d.parts[-1]
                ):
                continue
            
            # is directory empty?
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
                    self._handle_incomplete_run(d)
                    return d
            
            # load the metadata.yml file contained in d
            with open(d.joinpath("metadata.yml"), "r") as f:
                mdata = yaml.load(f, Loader=yaml.CLoader)
                if not self.model_name == mdata["model_name"]:
                    continue

            # are we using a dimensionality reduction model?
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
                self._verify_previous_embedding_directory(d)

    def _verify_previous_embedding_directory(self, d):
        """
        Check if number of embedding files and number of audio files match 
        to decide if this directory contains all the embeddings for the 
        current combination of model and audio dir. If the number of audio 
        files and the number of embedding files deviate by more than 1% 
        then continue with the missing files. If not treat the run as 
        complete and load the metadata and asign class attributs
        based on it.

        Parameters
        ----------
        d : None
        """
        try:
            num_files = len(
                [f for f in list(d.rglob(f"*{self.embed_suffix}"))]
            )
            num_audio_files = len(
                self.get_audio_files(self.audio_dir)
                )
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
            return

        if num_audio_files == num_files:
            self.combination_already_exists = True
            self._get_metadata_dict(d)
            logger.info(
                "\n### Embeddings already exist. "
                f"Using embeddings in {self.metadata_dict['embed_dir']} ###"
            )
            return
        elif (
            # allow 1 % deviation
            np.round(num_files / num_audio_files, 2) >= 0.99
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
            return
        else:
            self._handle_incomplete_run(d)

    def _handle_incomplete_run(self, directory):
        self.continue_incomplete_run = True
        self.embed_dir = directory
        self.files = self.get_audio_files(self.audio_dir)
        self._init_metadata_dict()
        self._get_metadata_from_created_embeddings()

    def _get_audio_paths_and_init_embed_dir(self):
        self.files = self.get_audio_files(self.audio_dir)
        self.files.sort()
        if not hasattr(self, 'embed_parent_dir'):
            from bacpipe import settings as bacpipe_settings
            self.embed_parent_dir = bacpipe_settings.embed_parent_dir
        self.embed_dir = (
            Path(self.embed_parent_dir)
            .joinpath(self._get_timestamp_dir())
            )

    def _get_annotation_files(self):
        all_annotation_files = list(self.audio_dir.rglob("*.csv"))
        audio_stems = [file.stem for file in self.files]
        self.annot_files = [
            file for file in all_annotation_files 
            if file.stem in audio_stems
        ]

    @staticmethod
    def get_audio_files(
        audio_dir, 
        audio_suffixes=bacpipe_settings.audio_suffixes,
        return_as='pathlib.Path'
        ):
        audio_dir = Path(audio_dir)
        files_list = []
        [
            [files_list.append(ll) for ll in audio_dir.rglob(f"*{string}")]
            for string in audio_suffixes
        ]
        files_list = np.unique(files_list).tolist()
        assert len(files_list) > 0, "No audio files found in audio_dir."
        if return_as == 'pathlib.Path':
            return files_list
        elif return_as == 'str':
            return [str(f) for f in files_list]
        

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

    def _get_embeddings(self):
        embed_dir = self.get_embedding_dir()
        self.files = [f for f in embed_dir.rglob(f"*{self.embed_suffix}")]
        self.files.sort()

        if not self.combination_already_exists:
            self._get_metadata_dict(embed_dir)
            self.metadata_dict["files"].update(
                {"embedding_files": [], "embedding_dimensions": []}
            )
            self.embed_dir = Path(self.dim_reduc_parent_dir).joinpath(
                self._get_timestamp_dir() + f"-{self.model_name}"
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

    def _get_timestamp_dir(self):
        if self.dim_reduction_model:
            model_name = self.dim_reduction_model
        elif not self.model_name:
            model_name = ''
        else:
            model_name = self.model_name
        return time.strftime(
            "%Y-%m-%d_%H-%M___" + model_name + "-" + self.audio_dir.stem,
            time.localtime(),
        )

    def read_embedding_file(self, file):
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
        if not self.files[0].suffix == self.embed_suffix:
            self.files = list(self.embed_dir.rglob(f'*{self.embed_suffix}'))
            self.files.sort()
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
            return np.vstack(list(d.values()))
        
    def predictions(self, threshold=bacpipe_settings.classifier_threshold, as_type='dict'):
        preds_path = (
            self.evaluations_dir 
            / self.model_name 
            / 'classification'
            / 'original_classifier_outputs'
            )
        import bacpipe
        self.embedder = bacpipe.Embedder(self.model_name, self)
        if not preds_path.exists():
            logger.warning(
                "No classifier predictions have been save yet. "
            )
            return None
        class_files = list(preds_path.rglob('*.json'))
        if not class_files:
            logger.warning(
                "No classifier predictions have been save yet. "
            )
            return None
        d = {}
        l2i = {label: idx for idx, label in enumerate(self.embedder.model.classes)}
        array_label_indices = []
        preds_array = np.array([], dtype=np.int8)
        
        from bacpipe.embedding_evaluation.label_embeddings import fill_all_labels_array
        
        for file in class_files:
            with open(file, "r") as f:
                preds = json.load(f)
            vals = np.ones([
                preds['head']['Time bins in this file'],
                len(preds)-1
                ]) * -1
            preds.pop('head')
            for idx, (class_label, val) in enumerate(preds.items()):
                array_label_indices.append(l2i[class_label])
                # vals[val['time_bins_exceeding_threshold'], idx] = val['classifier_predictions']
                vals[val['time_bins_exceeding_threshold'], idx] = l2i[class_label]
            
            preds_array = fill_all_labels_array(vals, preds_array)
            d[str(file.relative_to(preds_path))] = vals
            
        i2l = {v: k for k, v in l2i.items()}
        
        if as_type == 'dict':
            return d, i2l
        elif as_type == 'array':
            return preds_array, i2l

    def _write_audio_file_to_metadata(self, file, model, embeddings, file_length):
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
        # sort files in case a run was continued and files were added
        sorted_indices = np.argsort(self.metadata_dict["files"]["audio_files"])
        for key, lists in self.metadata_dict["files"].items():
            self.metadata_dict["files"][key] = np.array(lists)[sorted_indices].tolist()
        
        with open(str(self.embed_dir.joinpath("metadata.yml")), "w") as f:
            yaml.safe_dump(self.metadata_dict, f)

    def update_files(self):
        if self.dim_reduction_model:
            self.files = [
                f for f in self.embed_dir.iterdir() if f.suffix == ".json"
                ]
        else:
            self.files = list(self.embed_dir.rglob("*.npy"))
            
    def save_embedding_file(self, file, embeds):
        if self.dim_reduction_model:
            file_dest = self.embed_dir.joinpath(
                self.audio_dir.stem + "_" + self.model_name
            )
            file_dest = str(file_dest) + ".json"
            input_len = (
                self.metadata_dict["segment_length (samples)"]
                / self.metadata_dict["sample_rate (Hz)"]
            )
            self._save_embeddings_dict_with_timestamps(
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

    def _save_embeddings_dict_with_timestamps(
        self, file_dest, embeds, input_len
    ):
        t_stamps = []
        d = {
            var: embeds[:, i].tolist() 
            for i, var in zip(range(embeds.shape[1]), ["x", "y"])
        }
        
        embedding_dimensions = self.metadata_dict["files"]["embedding_dimensions"]

        for num_segments, *_ in embedding_dimensions:
            [
                t_stamps.append(np.round(t, 4)) 
                for t in np.arange(0, num_segments) * input_len
            ]
            
        d["timestamp"] = t_stamps

        d["metadata"] = {
            k: (v if isinstance(v, list) else v)
            for (k, v) in self.metadata_dict["files"].items()
        }
        
        d["metadata"].update(
            {
                k: v 
                for (k, v) in self.metadata_dict.items() 
                if not isinstance(v, dict)
            }
        )

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
            np.save(
                file_dest.replace(".json", f"{embeds.shape[-1]}.npy"), 
                embed_dict
                )
            
    def classifier_should_be_run(
        self, paths, run_pretrained_classifier, testing, **kwargs
        ):
        
        if (
            testing 
            or (
                not paths.preds_path.joinpath(
                "original_classifier_outputs"
                ).exists() 
                and run_pretrained_classifier
            )
        ):
            if self.model_name in ['perch_v2', 'perch_bird', 'vggish', 'surfperch', 'google_whale']:
                logger.warning(
                    f"\n \n The google family of models (which {self.model_name} is part of) "
                    "calculate embeddings and classifications at once, making it "
                    "impossible to only run the classifier, like with any other model. "
                    "Please remove the embeddings corresponding to this model and then "
                    "rerun bacpipe with the setting `run_pretrained_classifier` set to True. "
                    "That way classification results will be saved immediately.\n \n"
                )
                return False
            elif self.model_name in ['audioprotopnet']:
                logger.warning(
                    f"\n \n The {self.model_name} model requires spatial embeddings "
                    "due to its prototypical classification head. The embeddings that "
                    "bacpipe saves for consistency are always average pooled over the "
                    "entire input audio. Therefore the embeddings generated by this model "
                    "cannot be used for downstream classification. "
                    "Please remove the embeddings corresponding to this model and then "
                    "rerun bacpipe with the setting `run_pretrained_classifier` set to True. "
                    "That way classification results will be saved immediately.\n \n"
                )
                return False
            elif self.model_name in ['avesecho_passt']:
                logger.warning(
                    f"\n \n The {self.model_name} model "
                    "calculates embeddings and classifications at once, making it "
                    "impossible to only run the classifier, like with any other model. "
                    "Please remove the embeddings corresponding to this model and then "
                    "rerun bacpipe with the setting `run_pretrained_classifier` set to True. "
                    "That way classification results will be saved immediately.\n \n"
                )
                return False
            else:
                return True

def replace_default_kwargs_with_user_kwargs(remove_keys=None, **kwargs):
    from bacpipe import config, settings
    default_kwargs = {**vars(config), **vars(settings)}
    if remove_keys:
        for key in remove_keys:
            default_kwargs.pop(key)
    
    for k, v in kwargs.items():
        if k in default_kwargs:
            default_kwargs[k] = kwargs[k]
            
    replaced_kwargs = default_kwargs
    
    # if there are any other kwargs put them back in
    for k, v in kwargs.items():
        replaced_kwargs[k]= v
    return replaced_kwargs