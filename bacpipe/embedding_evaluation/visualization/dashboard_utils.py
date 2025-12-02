import panel as pn
import matplotlib
import json
import seaborn as sns
import numpy as np
import pandas as pd

import torch.nn.functional as F
import torch

from pathlib import Path
from bacpipe.embedding_evaluation.classification.train_classifier import LinearClassifier

sns.set_theme(style="whitegrid")

matplotlib.use("agg")

class DashBoardHelper:
    
    def init_plot(self, p_type, plot_func, widget_idx, **kwargs):
        getattr(self, f"{p_type}_plot")[widget_idx] = pn.panel(
            self.plot_widget(plot_func, widget_idx=widget_idx, **kwargs), tight=False
        )
        return getattr(self, f"{p_type}_plot")[widget_idx]

    def plot_widget(self, plot_func, **kwargs):
        if kwargs.get("return_fig", False):
            return pn.bind(plot_func, **kwargs)
        else:
            return self.add_save_button(plot_func, **kwargs)

    def widget(self, name, options, attr="Select", width=120, **kwargs):
        return getattr(pn.widgets, attr)(
            name=name, options=options, width=self.widget_width, **kwargs
        )

    def init_widget(self, idx, w_type, **kwargs):
        getattr(self, f"{w_type}_select")[idx] = self.widget(**kwargs)
        return getattr(self, f"{w_type}_select")[idx]
    
    def collect_all_embeddings(self, model):
        import bacpipe.generate_embeddings as ge
        dataset_path = self.path_func(model).dataset_path
        audio_dir = dataset_path.stem
        parent_dir = dataset_path.parent
        ld = ge.Loader(audio_dir=audio_dir, 
                       model_name=model, 
                       main_results_dir=parent_dir, 
                       **self.kwargs)
        
        for idx, file in enumerate(ld.files):
            if idx == 0:
                embeds = np.load(file)
            else:
                embeds = np.vstack([embeds, np.load(file)])
        return torch.Tensor(embeds)
    
    def run_classifier(self, embeds, linear_clfier):
        probs = []
        for idx, batch in enumerate(embeds):
            logits = linear_clfier(batch)
            probabilities = F.softmax(logits, dim=0).detach().cpu().numpy()
            binary_classification = np.zeros(probabilities.shape, dtype=np.int8)
            binary_classification[probabilities > self.class_threshold] = 1
            probs.append(binary_classification.tolist())
            self.progress_bar.value = int((idx+1)/len(embeds)*100)
        self.classifier_complete = True
        return np.array(probs, dtype=np.int8)
            
    
    def classify_embeddings(self, model, path, threshold, event):
        if path == '':
            path = (
                self.path_func(self.models[0]).class_path / 'linear_classifier.pt'
                ).as_posix()
        if threshold == '':
            self.class_threshold = 0.5
        else:
            self.class_threshold = float(threshold)
        
        embeds = self.collect_all_embeddings(model)
        
        with open(Path(path).parent / 'label2index.json', 'r') as f:
            label2index = json.load(f)
        self.species_select[0].options = list(label2index.keys())
            
        clfier_weights = torch.load(path)
        clfier = LinearClassifier(clfier_weights['clfier.weight'].shape[-1], len(label2index))
        clfier.load_state_dict(clfier_weights)
        clfier.to(self.kwargs['device'])
        
        probs = self.run_classifier(embeds, clfier)
        # self.class_figure
        self.class_tuples = probs, label2index
        return probs, label2index
        
    def get_timestamps(self, eval_dir, model, label_key):
        from datetime import datetime 
        default_labels = np.load(
            # self.ld.evaluations_dir 
            # / self.ld.metadata_dict['model_name']
            eval_dir
            / model
            / 'labels/default_labels.npy',
            allow_pickle=True
            ).item()
        labels = default_labels[label_key]
        if label_key == 'time_of_day':
            labels_ts = [datetime.strptime(ts ,'%H-%M-%S').timestamp() for ts in labels]
        if label_key == 'continuous_timestamp':
            labels_ts = [datetime.strptime(ts ,'%Y-%m-%d %H-%M-%S').timestamp() for ts in labels]
        return labels_ts
        
        
    def get_classes(self, path):
        if path == '':
            path = (
                self.path_func(self.models[0]).class_path / 'linear_classifier.pt'
                ).as_posix()
        with open(Path(path).parent / 'label2index.json', 'r') as f:
            classes = json.load(f)
        return list(classes.keys())
            

        
    def prepare_heatmap(self, model, progress, widget_idx=0):
        
        # timestamps = self.get_timestamps(self.path_func(model).eval_path, model, 'continuous_timestamp')
        if progress == False:
            return None
        
        binary_presence, class_dict = self.trigger_classification(progress)
        timestamps = self.get_timestamps_per_embedding(model)
        
        
        accumulated_presence = pn.bind(self.accumulate_data, 
                                       binary_presence, 
                                       timestamps, 
                                       class_dict,
                                       species=self.species_select[widget_idx],
                                       accumulate_by=self.accumulate_select[widget_idx])
        
        return self.plot_widget(self.plot_heatmap, 
                         accumulated_presence=accumulated_presence, 
                         timestamps=timestamps,
                         accumulate_by=self.accumulate_select[widget_idx], 
                         species=self.species_select[widget_idx])
        
        
    def plot_heatmap(self, accumulated_presence, timestamps, accumulate_by, species):
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=[10, 8])
        fig.suptitle(
            f'Presence heatmap for {species} with threshold of {self.class_threshold}'
            )
        sns.heatmap(accumulated_presence, 
                    vmin=0,
                    cmap='viridis')
        locs, labels = plt.xticks()
        if accumulate_by == 'day':
            labels = np.unique([ts.date() for ts in timestamps])
        plt.xticks(locs, labels, rotation=45)
        
        locs, labels = plt.yticks()
        plt.yticks(locs, labels, rotation=0)
        return fig      
    
    def accumulate_data(self, presence, timestamps, class_dict, species, accumulate_by='day'):
        species_idx = class_dict[species]
        dates = np.array([getattr(ts, 'date')() for ts in timestamps])
        hours = np.array([getattr(ts, 'hour') for ts in timestamps])
        accumulated = np.zeros([24, len(np.unique(dates))], dtype=np.int8)
        species_presence = presence[:, species_idx]
        for date_idx, date in enumerate(np.unique(dates)):
            daily_presence_idx = np.where(dates==date)[0]
            for hour in range(24):
                hourly_presence_idx = np.where(hours[daily_presence_idx]==hour)[0]
                accumulated[hour, date_idx] = sum(species_presence[daily_presence_idx[hourly_presence_idx]])
        return accumulated
            
        

    def get_timestamps_per_embedding(self, model):
        from bacpipe.embedding_evaluation.label_embeddings import DefaultLabels as DL
        import datetime as dt
        
        embed_dict = self.vis_loader.embeds[model]
        ts_within_audio_files = [dt.timedelta(seconds=ts) for ts in embed_dict['timestamp']]
        ts_files = [DL.get_dt_filename(f) for f in embed_dict['metadata']['audio_files']]
        ts_files_same_length_as_embeds = []
        [
            ts_files_same_length_as_embeds.extend([ts_file] * embed_len) 
            for ts_file, embed_len in zip(ts_files, embed_dict['metadata']['nr_embeds_per_file'])
        ]
        
        ts_per_embedding = [
            ts_file+ts_within_audio_file
            for ts_file, ts_within_audio_file in 
            zip(ts_files_same_length_as_embeds, ts_within_audio_files)
            ]
        return ts_per_embedding
