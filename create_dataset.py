import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import librosa
from tqdm import tqdm


import h5py

SEED = 42 # ensure that always the same context files get selected
GLOBAL_LENGTH = 3 # 5s is the standard for bird volcalizations
SR = 32_000
NR_REPITITIONS = 3
RATIO_NOISE_TO_TARGET = 1
PLOT = False
# src_path = '/media/siriussound/Extreme SSD/Recordings'
noise_srcs = {
    # 'anura': f'{src_path}/terrestrial/Amphibians/AnuranSet/AnuranSet',
    # 'wabad': f'{src_path}/terrestrial/Birds/WABAD',
    'audiomoth_leiden': '/home/siriussound/Code/identifying_unknown_species/data/context/just_noise',
    'AnuranSet_INCT41': '/mnt/swap/Work/Data/Amphibians/AnuranSet/AnuranSet/INCT41'
    }

target_paths = {
    'white-crested turaco': '/home/siriussound/Code/identifying_unknown_species/data/target_species/clean_target_sounds/birds/white-crested turaco/eq-ed and noise reduced',
    'Decticus albifrons': '/home/siriussound/Code/identifying_unknown_species/data/target_species/clean_target_sounds/insects/Decticus albifrons'
}

# RATIO_WITHIN_FILE = 2
# RATIO_DIFF_FILE = 1
PAD_FUNC = 'wrap'
USE_TUKEY_FILTER = True

# number of context files to copy to the get the contextual segments from
# NR_CNTXT_FILES = 50

np.random.seed(SEED)

import torch
import bacpipe
from bacpipe.core.audio_processor import AudioHandler

    

def load_audios(paths_dict):
    df = pd.DataFrame()
    target_audio = []
    for k, v in paths_dict.items():
        aud = AudioHandler(padding=PAD_FUNC, audio_dir=v, segment_length=GLOBAL_LENGTH*SR, sr=SR, device='cuda')
        
        audio_files = bacpipe.get_audio_files(v)

        for file in tqdm(audio_files, desc='load audio', total=len(audio_files)):
            raw_audio, sr = aud._load_and_resample(file)
            win_audio = aud._window_audio(raw_audio)
            win_audio = win_audio.cpu()
            target_audio.append(win_audio)
            df_tmp = pd.DataFrame()
            df_tmp['target'] = [k] * len(win_audio)
            df_tmp['file_stem'] = [file.stem] * len(win_audio)
            df_tmp['start'] = np.arange(len(win_audio)) * GLOBAL_LENGTH
            df_tmp['end'] = df_tmp['start'] + GLOBAL_LENGTH
            
            df = pd.concat([df, df_tmp])
    df.index = range(len(df))
            
    return target_audio, df



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
        rms_signal = np.mean(librosa.feature.rms(y=signal))
        rms_noise = np.mean(librosa.feature.rms(y=noise))

        # Calculate the required scaling factor for the noise
        # SNR = 20 * log10(rms_signal / (k * rms_noise))
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
                S = librosa.feature.melspectrogram(y=np.array(audio), sr=SR, n_mels=128,
                                                fmax=SR // 2)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, x_axis='time',
                                        y_axis='mel', sr=SR,
                                        fmax=SR // 2, ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set(title=string)
                if not string == 'mixed_audio':
                    ax.set_xticks([], [])
                    ax.set_xlabel('')
            fig.suptitle(f'{species}__{noise_env}__{snr_db}')
            path = Path('/home/siriussound/Code/identifying_unknown_species/data/figures')
            path_snr = path / f'snr_{snr_db}'
            path_snr_species = path_snr / species
            path_snr_species_noise = path_snr_species / noise_env
            path_snr_species_noise.mkdir(exist_ok=True, parents=True)
            fig.savefig(path_snr_species_noise / f'{idx}.png')
            plt.close(fig)
    return mixed_arrays


def collect_audio_segments():
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
    audios = []
    
    species_sounds, species_df = load_audios(target_paths)
    noise_sounds, noise_df = load_audios(noise_srcs)
    
    
    for noise_env in tqdm(noise_df.target.unique(), desc='noise_env', leave=False, position=0):
        for species in tqdm(species_df.target.unique(), desc='species', leave=False, position=1):
            species_idxs = species_df[species_df.target == species].index
            noise_idxs = noise_df[noise_df.target == noise_env].index
            
            species_audio = torch.vstack(species_sounds)[species_idxs]
            noise_audio = torch.vstack(noise_sounds)[noise_idxs]
            
            random_indices = np.random.permutation(len(noise_audio))
            
            for idx_rep, repetition in tqdm(enumerate(range(0, len(noise_audio), len(species_audio))), desc='rep', leave=False, position=2, total=NR_REPITITIONS):
                if idx_rep == NR_REPITITIONS:
                    break
                batch_random_indices = random_indices[repetition:repetition+len(species_audio)]
                rnd_noise_arrays = noise_audio[batch_random_indices]
                
                for snr in tqdm([0, 1.5, 3, 6, 9, 12], desc='snr', leave=False, position=3):
                    df_tmp = pd.DataFrame(data)
                    mixed = combined_target_and_noise(species_audio, rnd_noise_arrays, species=species, noise_env=noise_env, snr_db=snr, plot=PLOT)
                    
                    audios.append(torch.vstack(mixed))
                    
                    df_tmp.species_filename = species_df.file_stem[species_idxs][:len(mixed)]
                    df_tmp.noise_filename = noise_df.file_stem[noise_idxs[batch_random_indices]].values
                    df_tmp.species_start = species_df.start[species_idxs]
                    df_tmp.species_end = species_df.end[species_idxs]
                    df_tmp.noise_start = noise_df.start[noise_idxs[batch_random_indices]].values
                    df_tmp.noise_end = noise_df.end[noise_idxs[batch_random_indices]].values
                    df_tmp.species = species
                    df_tmp.noise_env = noise_env
                    df_tmp.snr = snr
                    
                    df = pd.concat([df, df_tmp])
            
        # these are pure noise to cluster against
        df_tmp = pd.DataFrame(data)
        
        remaining_random_indices = random_indices[repetition+len(species_audio):(repetition+len(species_audio))*RATIO_NOISE_TO_TARGET*NR_REPITITIONS]
        only_noise = noise_audio[remaining_random_indices]
        
        audios.append(only_noise)
        
        df_tmp.noise_filename = noise_df.file_stem[noise_idxs[remaining_random_indices]].values
        df_tmp.species_filename = ''
        df_tmp.species_start = -1
        df_tmp.species_end = -1
        df_tmp.noise_start = noise_df.start[noise_idxs[remaining_random_indices]].values
        df_tmp.noise_end = noise_df.end[noise_idxs[remaining_random_indices]].values
        df_tmp.species = ''
        df_tmp.noise_env = noise_env
        df_tmp.snr = -1
        
        df = pd.concat([df, df_tmp])
                
                
    audios = torch.vstack(audios)
    return audios, df



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
        if k in ['species', 'noise_env', 'species_filename', 'noise_filename']:
            file.create_dataset(k, data=df[k], dtype=dt)
        else:
            file.create_dataset(k, data=df[k])
        

    file.attrs["description"] = f"""
    Dataset of species vocalizations with superimposed noise environments. 
    Length of all sound segments = {GLOBAL_LENGTH}. 
    Global SR for all = {SR}.
    """
    
        
def create_dataset():
    audio, df = collect_audio_segments()
    
    file_name = f"unknown_sounds_len_{GLOBAL_LENGTH}_sr_{SR}_repetitions_{NR_REPITITIONS}_ratio-n2t_{RATIO_NOISE_TO_TARGET}"
    save_path = f"data/data_h5_files/{file_name}"
    with h5py.File(save_path + '.h5', "w") as f:
        write_dataset_to_file(f, audio, df)
    
    df.to_csv(save_path + '.csv', index=False)
        

def read_dataset(file):
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
        
        audio = data['audio'][:]
    
    print('loaded')
    return df, audio


if __name__ == '__main__':
    create_dataset()
    # df, audio = read_dataset()
            
            
    print('worked')