import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import librosa as lb
from tqdm import tqdm
import audioread

import h5py

SEED = 42 # ensure that always the same context files get selected
GLOBAL_LENGTH = 3 # 5s is the standard for bird volcalizations
SR = 32_000 # a lot of the data is sampled lower than that anyway
RATIO_NOISE_TO_TARGET = 50 # ratio between total target to total noise

# This is the number of vocalizations per species that is superimposed
# with each noise environment. so total number of vocalizations for one
# species will be this number * number of noise environments. So this
# is directly linked to the max cluster size that we define in the clustering
# step because if this is too high than in an ideal scenario our cluster would
# be too big to be identified. And we do want this value to be small to represent
# obscure rare classes.
NR_SEGMENTS_PER_SPECIES = 10

PLOT = True
# src_path = '/media/siriussound/Extreme SSD/Recordings'
noise_srcs = {
    # 'anura': f'{src_path}/terrestrial/Amphibians/AnuranSet/AnuranSet',
    # 'wabad': f'{src_path}/terrestrial/Birds/WABAD',
    # 'heijmans': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/context/just_noise/heijmans',
    'germany_campsite': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/context/just_noise/germany_campsite',
    'BIRB_NES': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/BirdSet/NES - neotropical coffee farms in Colombia and Costa Rica/soundscape_data/audio',
    'french_guyana': '/media/siriussound/Extreme SSD/Recordings/MNHN/darksound/dB@DARKSOUND/AUDIO/04662-I21',
    'Silencio': '/media/siriussound/Extreme SSD/Recordings/terrestrial/BirdClef',
    # 'audiomoth_leiden': '/media/siriussound/Extreme SSD/Recordings/MyRecordings/20250701_AudioMothsLeiden/A1_NW_24E1440360369142/20250605/nighttime',
    'audiomoth_leiden': '/media/siriussound/Extreme SSD/Recordings/MyRecordings/20250701_AudioMothsLeiden/A1_NW_24E1440360369142',
    ### AnuraSet is excluded because all of my amphibian species are now from there!
    # 'AnuranSet_INCT41': '/mnt/swap/Work/Data/Amphibians/AnuranSet/AnuranSet/INCT41'
    }

target_paths = {
    
    ### Birds
    'white-crested turaco': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/birds/white-crested turaco/edited',
    'tiny cisticola': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/birds/tiny cisticola/edited', 
    'rufous-crowned roller': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/birds/rufous-crowned roller/edited', 
    
    ### Insects
    'Acrometopa servillea': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/insects/Acrometopa servillea/edited',
    'Oecanthus Dulcisonans': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/insects/Oecanthus Dulcisonans/edited',
    'Svercus Palmetorum': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/insects/Svercus Palmetorum/edited',
    
    ### Amphibians
    'Adenomera Marmorata': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/amphibians/Adenomera Marmorata/edited',
    'Dendropsophus Cruzi': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/amphibians/Dendropsophus Cruzi/edited',
    'Scinax Fuscomarginatus': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/amphibians/Scinax Fuscomarginatus/edited',
    
    ### Mammals
    'Agile Gibbon': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/mammals/Agile Gibbon/edited',
    'Arctic Fox': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/mammals/Arctic Fox/edited',
    'Neotine Giant Otters': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/mammals/Neotine Giant Otters/edited',
    
    # 'Decticus albifrons': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/insects/Decticus albifrons/eq-ed and noise reduced',
    # 'Schmidts Marbled Bush-cricket': '/mnt/swap/Work/Data/identifying_unknown_sounds_data/data/target_species/clean_target_sounds/insects/Schmidts Marbled Bush-cricket/eq-ed and noise reduced'
}


main_path = Path('/media/siriussound/Extreme SSD/identifying_unknown_sounds')
# main_path = Path('/mnt/swap/Work/Data/identifying_unknown_sounds_data/data')

# RATIO_WITHIN_FILE = 2
# RATIO_DIFF_FILE = 1
PAD_FUNC = 'minimum'
USE_TUKEY_FILTER = True

# number of context files to copy to the get the contextual segments from
# NR_CNTXT_FILES = 50

np.random.seed(SEED)

import torch
import bacpipe
from bacpipe.core.audio_processor import AudioHandler

def get_noise_df(paths_dict):
    df = pd.DataFrame()
    
    for k, v in paths_dict.items():
        
        audio_files = bacpipe.get_audio_files(v)
        if k == 'audiomoth_leiden':
            audio_files = [
                f for f in audio_files
                if (
                    f.stem.split('_')[-1][0] == '0'
                    and f.stem.split('_')[1][-3] == '5'
                    )
            ]

        for file in tqdm(audio_files, desc='get file lengths', total=len(audio_files)):
            
            with audioread.audio_open(file) as f:
                length = f.duration
            frames = int(np.ceil(length / (GLOBAL_LENGTH)))
            df_tmp = pd.DataFrame()
            df_tmp['target'] = [k] * frames
            df_tmp['file_stem'] = [file.stem] * frames
            
            df_tmp['start'] = np.arange(frames) * GLOBAL_LENGTH
            
            df_tmp['end'] = df_tmp['start'] + GLOBAL_LENGTH  
            df = pd.concat([df, df_tmp])
    df.index = range(len(df))
    return df

def load_noise(species_df):
    random_indices = {}
    df_noise = pd.DataFrame()
    
    for noise_env in tqdm(noise_srcs.keys(), desc='noise_env', leave=False, position=1):
        
        
        file_name = f"unknown_sounds_len_{GLOBAL_LENGTH}_sr_{SR}_nr-target_{NR_SEGMENTS_PER_SPECIES}_ratio-n2t_{RATIO_NOISE_TO_TARGET}_{noise_env}"
        save_path = main_path / f"data_h5_files/{NR_SEGMENTS_PER_SPECIES}_ratio-n2t_{RATIO_NOISE_TO_TARGET}/{file_name}.h5"
        if not save_path.exists():        
            noise_df = get_noise_df({noise_env: noise_srcs[noise_env]})
            sum_species_df_len = sum([len(species_df[species_df.target == species]) for species in species_df.target.unique()])
            noise_idxs = noise_df[noise_df.target == noise_env].index
            noise = []
            
            total_number_noise_segments = sum_species_df_len * (RATIO_NOISE_TO_TARGET / len(noise_srcs.keys())) 
            random_indices[noise_env] = np.random.permutation(len(noise_idxs.values))[:int(total_number_noise_segments)]
            this_env_noise_df = noise_df.loc[noise_idxs].iloc[random_indices[noise_env]]
            
            aud = AudioHandler(
                padding=PAD_FUNC, 
                audio_dir=noise_srcs[noise_env], 
                segment_length=GLOBAL_LENGTH*SR, 
                sr=SR, 
                only_embed_annotations=True
                )
            files = bacpipe.get_audio_files(noise_srcs[noise_env])
            selected_files = [f for f in files if f.stem in this_env_noise_df.file_stem.unique()]
            rand_order = np.random.permutation(len(selected_files))
            selected_files = np.array(selected_files)[rand_order]
            
            df_cumulative_noise = pd.DataFrame()
            for file in tqdm(
                selected_files,
                'Loading noise segments',
                total=len(selected_files),
                position=3
                ):
                tmp_df = this_env_noise_df[this_env_noise_df.file_stem == file.stem]
                frames, sr = aud.return_windowed_audio(file, annotations_df=tmp_df)
                noise.append(frames)        
                
                df_cumulative_noise = pd.concat([df_cumulative_noise, tmp_df])
            
            
            data = {
                'species': [],
                'noise_env': [],
                'snr': [],
                'species_filename': [],
                'noise_filename': [],
                'species_start': [],
                'species_end': [],
                'noise_start': [],
                'noise_end': [],
            }
            df_tmp_noise = pd.DataFrame(data)
            noise = torch.vstack(noise)
            
            df_tmp_noise.noise_filename = df_cumulative_noise.file_stem
            df_tmp_noise.noise_start = df_cumulative_noise.start
            df_tmp_noise.noise_end = df_cumulative_noise.end
            df_tmp_noise.noise_env = noise_env
            df_tmp_noise.species_filename = ''
            df_tmp_noise.species_start = -1
            df_tmp_noise.species_end = -1
            df_tmp_noise.species = ''
            df_tmp_noise.snr = -1
            
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            df_tmp_noise.to_csv(str(save_path).split('.')[0]+'.csv', index=False)
            with h5py.File(str(save_path), "w") as f:
                write_dataset_to_file(f, noise, df_tmp_noise)
            
        else:
            # df_tmp_noise, noise = read_dataset(file=save_path, return_audio=False)
            df_tmp_noise = pd.read_csv(str(save_path).split('.')[0]+'.csv', index_col=False)
        df_noise = pd.concat([df_noise, df_tmp_noise])
        
    return df_noise

def load_audios(paths_dict):
    df = pd.DataFrame()
    target_audio = []
    for k, v in paths_dict.items():
        species_audio = []
        df_species = pd.DataFrame()
        aud = AudioHandler(padding=PAD_FUNC, audio_dir=v, segment_length=GLOBAL_LENGTH*SR, sr=SR, device='cuda')
        
        audio_files = bacpipe.get_audio_files(v)

        for file in tqdm(audio_files, desc='load audio', total=len(audio_files)):
            raw_audio, sr = aud._load_and_resample(file)
            
            input_length = SR * GLOBAL_LENGTH
            
            rand_timeshift_offset = np.random.randint(input_length // 2)
            rand_shifted_audio = raw_audio[:, rand_timeshift_offset:]
            
            win_audio = aud._window_audio(rand_shifted_audio)
            win_audio = win_audio.cpu()
            species_audio.append(win_audio)
            df_tmp = pd.DataFrame()
            df_tmp['target'] = [k] * len(win_audio)
            df_tmp['file_stem'] = [file.stem] * len(win_audio)
            
            df_tmp['start'] = np.arange(len(win_audio)) * GLOBAL_LENGTH + rand_timeshift_offset/SR
            
            df_tmp['end'] = df_tmp['start'] + GLOBAL_LENGTH
            
            df_species = pd.concat([df_species, df_tmp])
                
        
        rnd_indices = np.random.permutation(len(df_species))[:NR_SEGMENTS_PER_SPECIES]
        
        df_species = df_species.iloc[rnd_indices]
        species_audio = torch.vstack(species_audio)[rnd_indices]
        
        
        df = pd.concat([df, df_species])
        target_audio.append(species_audio)
        
    df.index = range(len(df))
            
    return torch.vstack(target_audio), df



def combined_target_and_noise(
    signal_arrays, 
    noise_arrays, 
    snr_db, 
    species=None,
    noise_env=None,
    plot=False
    ):
    mixed_arrays = []
    for idx, (signal, noise) in tqdm(
        enumerate(zip(signal_arrays, noise_arrays)),
        desc='combining audio',
        total=len(signal_arrays),
        leave=False,
        position=4
        ):
        rms_signal = np.mean(lb.feature.rms(y=signal))
        rms_noise = np.mean(lb.feature.rms(y=noise))

        k = rms_signal / (rms_noise * (10 ** (snr_db / 20.0)))

        # Mix the signals
        mixed_audio = signal + (k * noise)
        mixed_arrays.append(mixed_audio)

        if plot and idx in np.random.permutation(len(signal_arrays))[:2]:
            # sanity check spectrogram
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=[10, 8])
            axes = fig.subplots(3, 1)  
            for ax, audio, string in zip(axes, [signal, noise, mixed_audio], ['signal', 'noise', 'mixed_audio']):
                S = lb.feature.melspectrogram(y=np.array(audio), sr=SR, n_mels=128,
                                                fmax=SR // 2)
                S_dB = lb.power_to_db(S, ref=np.max)
                img = lb.display.specshow(S_dB, x_axis='time',
                                        y_axis='mel', sr=SR,
                                        fmax=SR // 2, ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set(title=string)
                if not string == 'mixed_audio':
                    ax.set_xticks([], [])
                    ax.set_xlabel('')
            fig.suptitle(f'{species}__{noise_env}__{snr_db}')
            path_snr = main_path / f'data/figures/snr_{snr_db}'
            path_snr_species = path_snr / species
            path_snr_species_noise = path_snr_species / noise_env
            path_snr_species_noise.mkdir(exist_ok=True, parents=True)
            fig.savefig(path_snr_species_noise / f'{idx}.png')
            plt.close(fig)
    return mixed_arrays

def get_noise_segments(noise_env, batch_random_indices):
    
    file_name = f"unknown_sounds_len_{GLOBAL_LENGTH}_sr_{SR}_nr-target_{NR_SEGMENTS_PER_SPECIES}_ratio-n2t_{RATIO_NOISE_TO_TARGET}_{noise_env}"
    file = main_path / f"data_h5_files/{NR_SEGMENTS_PER_SPECIES}_ratio-n2t_{RATIO_NOISE_TO_TARGET}/{file_name}.h5"
    with h5py.File(file, 'r') as data:
        audio = data['audio'][batch_random_indices]
        
    return audio

def build_audio_and_df(species_sounds, species_df, noise_df, snr):

    data = {
        'species': [],
        'noise_env': [],
        'snr': [],
        'species_filename': [],
        'noise_filename': [],
        'species_start': [],
        'species_end': [],
        'noise_start': [],
        'noise_end': [],
    }
    df = pd.DataFrame(data)
    
    species_augmented = []
    for noise_env in tqdm(noise_df.noise_env.unique(), desc='noise_env', leave=False, position=1):
        for species in tqdm(species_df.target.unique(), desc='species', leave=False, position=2):
            species_idxs = species_df[species_df.target == species].index
            tmp_noise_df = noise_df[noise_df.noise_env == noise_env]
            tmp_noise_df.index = range(len(tmp_noise_df))
            
            species_audio = species_sounds[species_idxs]
            
            indices = tmp_noise_df.index
            rand_indices = np.random.permutation(indices)
            batch_random_indices = rand_indices[:len(species_audio)]
            batch_random_indices.sort()
            rnd_noise_arrays = get_noise_segments(noise_env, batch_random_indices)
        
            df_tmp = pd.DataFrame(data)
            mixed = combined_target_and_noise(species_audio, rnd_noise_arrays, species=species, noise_env=noise_env, snr_db=snr, plot=PLOT)
            
            species_augmented.append(torch.vstack(mixed))
            
            df_tmp.species_filename = species_df.file_stem[species_idxs][:len(mixed)]
            df_tmp.noise_filename = tmp_noise_df.noise_filename.iloc[batch_random_indices].values
            df_tmp.species_start = species_df.start[species_idxs]
            df_tmp.species_end = species_df.end[species_idxs]
            df_tmp.noise_start = tmp_noise_df.noise_start.iloc[batch_random_indices].values
            df_tmp.noise_end = tmp_noise_df.noise_end.iloc[batch_random_indices].values
            df_tmp.species = species
            df_tmp.noise_env = noise_env
            df_tmp.snr = snr
            
            df = pd.concat([df, df_tmp])
                
    species_augmented = torch.vstack(species_augmented)
    return species_augmented, df
        
def check_audio(audio):
    
    import sounddevice as sd
    import h5py
    from pathlib import Path
    import matplotlib.pyplot as plt
    import librosa as lb
    import numpy as np
    
    plt.figure(figsize=[10, 8])
    
    SR1 = 32_000
    SR = 48_000
    sd.play(audio, samplerate=SR1)
    S = lb.feature.melspectrogram(y=np.array(audio), sr=SR, n_mels=128,
                                    fmax=SR // 2)
    S_dB = lb.power_to_db(S, ref=np.max)
    img = lb.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=SR,
                            fmax=SR // 2)
    plt.colorbar(img, format='%+2.0f dB')
    plt.savefig(f'7.png')
    plt.close()


def write_dataset_to_file(file, audio, df, chunk_size=500):
    # --- Audio ---
    # audio_data = np.array(data['audio'])
    n_samples = len(df)
    audio = np.array(audio)
    shape = audio.shape
    dtype = audio.dtype

    dset = file.create_dataset(
        "audio",
        shape=shape,
        dtype=dtype,
        chunks=(min(chunk_size, n_samples),) + shape[1:]
    )

    for i in range(0, n_samples, chunk_size):
        dset[i:i+chunk_size] = audio[i:i+chunk_size]

    # --- Metadata ---
    dt = h5py.string_dtype(encoding="utf-8")
    for k in df.columns:
        if k in ['species', 'target', 'noise_env', 'species_filename', 'noise_filename']:
            file.create_dataset(k, data=df[k], dtype=dt)
        else:
            file.create_dataset(k, data=df[k])
        

    file.attrs["description"] = f"""
    Dataset of species vocalizations with superimposed noise environments. 
    Length of all sound segments = {GLOBAL_LENGTH}. 
    Global SR for all = {SR}.
    """

        
def create_dataset():
    
    species_sounds, species_df =  load_audios(target_paths)
    noise_df = load_noise(species_df)
    
    for snr in tqdm([0, 1.5, 3, 6, 9, 12], desc='snr', leave=False, position=0):
        file_name = f"unknown_sounds_len_{GLOBAL_LENGTH}_sr_{SR}_nr-target_{NR_SEGMENTS_PER_SPECIES}_ratio-n2t_{RATIO_NOISE_TO_TARGET}_{snr=}"
        save_path = main_path / f"data_h5_files/{NR_SEGMENTS_PER_SPECIES}_ratio-n2t_{RATIO_NOISE_TO_TARGET}/{file_name}.h5"
        
        if not save_path.exists():
            species_augmented, df = build_audio_and_df(species_sounds, species_df, noise_df, snr)
            
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            with h5py.File(str(save_path), "w") as f:
                write_dataset_to_file(f, species_augmented, df)
            
            df.to_csv(str(save_path) + '.csv', index=False)
            del species_augmented, df
        
        

def read_dataset(file, return_audio=True):
    df_columns = {
        'species': [],
        'noise_env': [],
        'snr': [],
        'species_filename': [],
        'noise_filename': [],
        'species_start': [],
        'species_end': [],
        'noise_start': [],
        'noise_end': [],
    }
    df = pd.DataFrame(df_columns)
    
    with h5py.File(file, 'r') as data:
        for k in df_columns.keys():
            if k in ['species', 'noise_env', 'species_filename', 'noise_filename']:
                df[k] = data[k][:].astype(str)
            else:
                df[k] = data[k][:]
        if return_audio:
            audio = data['audio'][:]
        else:
            audio = None
    
    # verify that df is corect
    df_from_csv = pd.read_csv(str(file).replace('.h5', '.csv'))
    df_from_csv = df_from_csv.fillna('')
    
    pd.testing.assert_frame_equal(df, df_from_csv)
    
    print('loaded')
    return df, audio


if __name__ == '__main__':
    create_dataset()
    # src = '/media/siriussound/Extreme SSD/identifying_unknown_sounds/data_h5_files/6_ratio-n2t_10'
    # file = 'unknown_sounds_len_3_sr_32000_repetitions_6_ratio-n2t_10_snr=0.h5'
    # df, audio = read_dataset(Path(src) / file)
            
            
    print('worked')