import librosa as lb
import numpy as np
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import logging
import importlib
logger = logging.getLogger('bacpipe')

class Loader():
    def __init__(self, check_if_combination_exists=True, 
                 model_name='umap', **kwargs):
        self.model_name = model_name
        
        with open('bacpipe/config.yaml', "r") as f:
            self.config =  yaml.safe_load(f)
            
        for key, val in self.config.items():
            setattr(self, key, val)
        
        self.check_if_combination_exists = check_if_combination_exists
        if self.model_name == 'umap':
            self.embed_suffix = '.json'
        else:
            self.embed_suffix = '.npy'
            
        self.check_embeds_already_exist()
        if self.combination_already_exists or self.model_name == 'umap':
            self.get_embeddings()
        else:
            self._get_audio_paths()
            self._init_metadata_dict()
        
        if not self.combination_already_exists:
            self.embed_dir.mkdir(exist_ok=True, parents=True)
        else:
            if self.model_name == 'umap':
                self.embed_dir = self.umap_embed_dir
            logger.debug(
                'Combination of {} and {} already '
                'exists -> using saved embeddings in {}'
                .format(self.model_name,
                        Path(self.audio_dir).stem,
                        str(self.embed_dir))
                )
        
    def check_embeds_already_exist(self):
        self.combination_already_exists = False
        self.umap_embed_dir = False
        
        if self.check_if_combination_exists:
            if self.model_name == 'umap':
                existing_embed_dirs = list(Path(self.umap_parent_dir)
                                           .iterdir())
            else:
                existing_embed_dirs = list(Path(self.embed_parent_dir)
                                           .iterdir())
            if isinstance(self.check_if_combination_exists, str):
                existing_embed_dirs = [existing_embed_dirs[0].parent
                                       .joinpath(
                                           self.check_if_combination_exists
                                       )]
            existing_embed_dirs.sort()
            for d in existing_embed_dirs[::-1]:
                
                if (self.model_name in d.stem 
                    and Path(self.audio_dir).stem in d.stem
                    and self.embedding_model in d.stem):
                    
                    num_files = len([f for f in d.iterdir() 
                                    if f.suffix == self.embed_suffix])
                    num_audio_files = len([
                        f for f in Path(self.audio_dir).iterdir()
                        if f.suffix in self.config['audio_suffixes']
                        ])
                    
                    if num_audio_files == num_files:
                        self.combination_already_exists = True
                        self._get_metadata_dict(d)
                        break
        
    def _get_audio_paths(self):
        self.audio_dir = Path(self.audio_dir)

        self.files = self._get_audio_files()
        
        self.embed_dir = (Path(self.embed_parent_dir)
                          .joinpath(self.get_timestamp_dir()))

    def _get_audio_files(self):
        files_list = []
        [[files_list.append(ll) for ll in self.audio_dir.rglob(f'*{string}')] 
         for string in self.config['audio_suffixes']]
        return files_list
    
    def _init_metadata_dict(self):
        self.metadata_dict = {
            'model_name': self.model_name,
            'sr': self.sr,
            'audio_dir': str(self.audio_dir),
            'embed_dir': str(self.embed_dir),
            'files' : {
                'audio_files': [],
                'file_lengths (s)': [],
            }
        }
        
    def _get_metadata_dict(self, folder):
        with open(folder.joinpath('metadata.yml'), "r") as f:
            self.metadata_dict =  yaml.safe_load(f)
        for key, val in self.metadata_dict.items():
            if isinstance(val, str) and Path(val).is_dir():
                setattr(self, key, Path(val))
        if self.model_name == 'umap':
            self.umap_embed_dir = folder
        
    def get_embeddings(self):
        embed_dir = self.get_embedding_dir()
        self.files = [f for f in embed_dir.iterdir() 
                      if f.suffix == self.embed_suffix]
        self.files.sort()
        if not self.combination_already_exists:
            self._get_metadata_dict(embed_dir)
            self.metadata_dict['files'].update(
                {'embedding_files': [],
                'embedding_dimensions': []}
            )
            self.embed_dir = (Path(self.umap_parent_dir)
                                .joinpath(self.get_timestamp_dir()
                                        + f'-{self.embedding_model}'))

    def get_embedding_dir(self):
        if self.model_name == 'umap':
            if self.combination_already_exists:
                self.embed_parent_dir = Path(self.umap_parent_dir)
            else:
                self.embed_parent_dir = Path(self.embed_parent_dir)
                self.embed_suffix = '.npy'
        else:
            return self.embed_dir
        self.audio_dir = Path(self.audio_dir)
        
        if self.umap_embed_dir:
            # check if they are compatible
            return self.umap_embed_dir
        
        embed_dirs = [d for d in self.embed_parent_dir.iterdir()
                    if self.audio_dir.stem in d.stem and 
                    self.embedding_model in d.stem]
        # check if timestamp of umap is after timestamp of model embeddings
        embed_dirs.sort()
        most_recent_emdbed_dir = embed_dirs[-1]
        return most_recent_emdbed_dir
    
    def get_annotations(self):
        pass
        
    def get_timestamp_dir(self):
        return time.strftime('%Y-%m-%d_%H-%M___'
                                      + self.model_name
                                      + '-'
                                      + self.audio_dir.stem,
                                      time.localtime())
        
    def embed_read(self, file):
        embeds = np.load(file)
        self.metadata_dict['files']['embedding_files'].append(
            str(file)
            )
        self.metadata_dict['files']['embedding_dimensions'].append(
            embeds.shape
            )
        return embeds, False
    
    def load(self, file):
        if not self.model_name in ['umap', 'tsne']:
            return self.audio_read(file)
        else:
            return self.embed_read(file)
    
    def audio_read(self, file):
        if not file.suffix in self.config['audio_suffixes']:
            logger.warning(f'{str(file)} could not be read due to unsupported format.')
            return None
        with open(file, 'rb') as r:
            audio, sr = lb.load(r)
        
        self.metadata_dict['files']['audio_files'].append(
            file.stem + file.suffix
            )
        self.metadata_dict['files']['file_lengths (s)'].append(
            len(audio)//sr
            )
        
        return audio, sr
    
    def write_metadata_file(self):
        with open(str(self.embed_dir.joinpath('metadata.yml')), 'w') as f:
            yaml.safe_dump(self.metadata_dict, f)
            
    def update_files(self):
        if self.model_name == 'umap':
            self.files = [f for f in self.embed_dir.iterdir() 
                          if f.suffix == '.json']

class Embedder:
    def __init__(self, model_name, **kwargs):
        import yaml
        with open('bacpipe/config.yaml', 'rb') as f:
            self.config = yaml.safe_load(f)
            
        self.model_name = model_name
        self._init_model()

    def _init_model(self):
        module = importlib.import_module(
            f'bacpipe.pipelines.{self.model_name}'
            )
        self.model = module.Model()

    def get_embeddings_from_model(self, input_tup):
        samples = self.model.preprocess(self.model.resample(input_tup))
        start = time.time()
        
        embeds = self.model(samples)
        if not isinstance(embeds, np.ndarray):
            embeds = embeds.numpy()
        
        logger.debug(f'{self.model_name} embeddings have shape: {embeds.shape}')
        logger.info(f'{self.model_name} inference took {time.time()-start:.2f}s.')
        return embeds

    def save_embeddings(self, file_idx, fileloader_obj, file, embeds):
        file_dest = fileloader_obj.embed_dir.joinpath(file.stem 
                                                      + '_'
                                                      + self.model_name)
        if file.suffix == '.npy':
            file_dest = str(file_dest) + '.json'
            input_len = 3
            save_embeddings_dict_with_timestamps(file_dest, embeds, input_len, 
                                                 fileloader_obj, file_idx)
            # TODO save png of embeddings for umap embeds
        else:
            file_dest = str(file_dest) + '.npy'
            np.save(file_dest, embeds)
        
def save_embeddings_dict_with_timestamps(file_dest, embeds, input_len, 
                                         loader_obj, f_idx):
    length = embeds.shape[0]
    lin_array = np.arange(0, length*input_len, input_len)
    d = {var: embeds[:, i].tolist() 
         for i, var in zip(range(embeds.shape[1]), ['x', 'y'])}
    d['timestamp'] = lin_array.tolist()
    
    d['metadata'] = {k: (v[f_idx] if isinstance(v, list) else v) 
                     for (k, v) in loader_obj.metadata_dict['files'].items()}
    d['metadata'].update({k: v for (k, v) in loader_obj.metadata_dict.items()
                          if not isinstance(v, dict)})
    
    import json
    with open(file_dest, 'w') as f:
        json.dump(d, f)
    
def generate_embeddings(save_files=True, **kwargs):
    ld = Loader(**kwargs)
    if not ld.combination_already_exists:    
        embed = Embedder(**kwargs)
        for idx, file in tqdm(enumerate(ld.files)):
            input_tup = ld.load(file)
            if input_tup is None:
                continue
            embeds = embed.get_embeddings_from_model(input_tup)
            embed.save_embeddings(idx, ld, file, embeds)
        ld.write_metadata_file()
        ld.update_files()
    return ld
