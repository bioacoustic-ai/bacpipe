import torch
from transformers import pipeline
import librosa as lb
from .utils import ModelBaseClass
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 48_000
LENGTH_IN_SAMPLES = 480_000

# figure out how to get the length right

# download the model so it does not need to be downloaded every time

class Model(ModelBaseClass):
    def __init__(self):
        super().__init__()
        
        self.audio_classifier = pipeline(
            task="zero-shot-audio-classification", 
            model="davidrrobinson/BioLingual"
            )
        self.model = self.audio_classifier.model.get_audio_features
    
    def preprocess(self, audio):
        input_tensor_shape = [1, 1, 1001, 64]
        re_audio = lb.resample(audio, 
                        orig_sr = self.config['sr'], 
                        target_sr = SAMPLE_RATE)
        num = np.ceil(len(re_audio) / LENGTH_IN_SAMPLES)
        # zero pad in case the end is reached
        re_audio = [*re_audio, 
                    *np.zeros([int(num * LENGTH_IN_SAMPLES - len(re_audio))])]
        wins = np.array(re_audio).reshape([int(num), LENGTH_IN_SAMPLES])
        all_wins = []
        
        print('Extracting preliminary features')
        for win in tqdm(wins):
            features = self.audio_classifier.feature_extractor(win,
                                                    sampling_rate = SAMPLE_RATE)
            aud_features = features['input_features'][0]
            aud_input = torch.tensor(aud_features.reshape(input_tensor_shape))
            all_wins.append(aud_input)
        all_wins = torch.stack(all_wins)
        return all_wins
    
    def __call__(self, input):
        print('Computing embeddings')
        embeds = []
        for sample in tqdm(input):
            embeds.append(self.model(sample).detach().numpy())
        return np.array(embeds).squeeze()
    