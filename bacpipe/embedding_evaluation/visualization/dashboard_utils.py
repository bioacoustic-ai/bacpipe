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
        import bacpipe.main as ge
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
                
            self.progress_bar.value = int((idx+1)/len(ld.files)*100)
        return torch.Tensor(embeds)
    
    def run_classifier(self, embeds, linear_clfier, threshold):
        probs = []
        embeds = embeds.to(self.kwargs['device'])
        for idx, batch in enumerate(embeds):
            logits = linear_clfier(batch)
            probabilities = F.softmax(logits, dim=0).detach().cpu().numpy()
            binary_classification = np.zeros(probabilities.shape, dtype=np.int8)
            binary_classification[probabilities > threshold] = 1
            probs.append(binary_classification.tolist())
            self.progress_bar.value = int((idx+1)/len(embeds)*100)
        self.classifier_complete = True
        return np.array(probs, dtype=np.int8)
            
    @staticmethod
    def verify_threshold(threshold):
        if threshold == '':
            threshold = 0.5
        else:
            threshold = float(threshold)
        return threshold
        
    
    def classify_embeddings(self, model, path, threshold, event):
        if path == '':
            path = (
                self.path_func(self.models[0]).class_path / 'linear_classifier.pt'
                ).as_posix()
        threshold = self.verify_threshold(threshold)
        
        embeds = self.collect_all_embeddings(model)
        
        self.loading_test_placeholder.value = 'Running classifier'
        
        with open(Path(path).parent / 'label2index.json', 'r') as f:
            label2index = json.load(f)
            
        clfier_weights = torch.load(path, map_location=self.kwargs['device'])
        clfier = LinearClassifier(clfier_weights['clfier.weight'].shape[-1], len(label2index))
        clfier.load_state_dict(clfier_weights)
        clfier.to(self.kwargs['device'])
        
        probs = self.run_classifier(embeds, clfier, threshold)
        
        return probs, label2index
    
    @staticmethod
    def reorder_by_most_occurrance(probs, label2index):
        sums = [sum(probs[:,a]) for a in range(probs.shape[1])]
        
        sorted_l2i = dict(sorted(
            label2index.items(), 
            key=lambda x: sums[x[1]],
            reverse=True
            ))
        return sorted_l2i
        
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
            
    def load_classification(self, model, thresh):
        integrated_clfier_path = (
            self.path_func(model)
            .class_path.joinpath('original_classifier_outputs')
        )
        if not integrated_clfier_path.exists():
            return None, None
        else:
            files = list(integrated_clfier_path.rglob('*json'))
            
        cl_dict = {}
        total_length = 0
        k2idx = {}
        for file in files:
            with open(file, 'r') as f:
                d = json.load(f)
                current_time_bins = d['head']['Time bins in this file']
                d.pop('head')
                
                for k, v in d.items():
                    cl_dict[k] = np.zeros([total_length + current_time_bins])    
                    if not k2idx:
                        k2idx[k] = 0
                    if not k in k2idx:
                        k2idx[k] = max(k2idx.values()) + 1
                        
                    cl_dict[k][np.array(v['time_bins_exceeding_threshold']) + total_length] = v['classifier_predictions']
                    # file_specific_classification[v['time_bins_exceeding_threshold'], k2idx[k]] = v['classifier_predictions']
                for species in [
                    k for k, v in cl_dict.items() 
                    if len(v) < total_length + current_time_bins
                    ]:
                    cl_dict[species] = np.hstack([cl_dict[species], np.zeros([current_time_bins])])
                
                total_length += current_time_bins
        
        probs_array = np.array(list(cl_dict.values())).T
        # binary_classification = probs_array[probs_array > thresh]
        binary_classification = np.zeros(probs_array.shape, dtype=np.int8)
        binary_classification[probs_array > thresh] = 1
        
        return binary_classification, k2idx

        
    def prepare_heatmap(self, threshold, model, clfier_type, progress, widget_idx=0):
        
        # timestamps = self.get_timestamps(self.path_func(model).eval_path, model, 'continuous_timestamp')
        if progress == False:
            return None
        threshold = self.verify_threshold(threshold)
        
        if clfier_type == 'Linear':
            self.loading_test_placeholder.value = 'Loading embeddings'
            binary_presence, class_dict = self.trigger_classification(progress)
        elif clfier_type == 'Integrated':
            binary_presence, class_dict = self.load_classification(model, threshold)
        
        
        if binary_presence is None:
            return pn.widgets.StaticText(
                name="Error",
                value="It seems like the classifier hasn't been run yet. Please rerun bacpipe with the setting "
                "`run default classifier` set to `True`."
                )
        
        timestamps = self.get_timestamps_per_embedding(model)
        
        class_dict['overall'] = len(class_dict)
        binary_presence = np.concatenate([binary_presence.T, 
                                     [np.sum(binary_presence, axis=1).astype(np.int8)]]).T
        
        
        class_dict = self.reorder_by_most_occurrance(binary_presence, class_dict)
        self.species_select[0].options = list(class_dict.keys())
        
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
                         species=self.species_select[widget_idx],
                         threshold=threshold)
        
        
    def plot_heatmap(self, accumulated_presence, timestamps, accumulate_by, species, threshold):
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=[11, 8])
        fig.suptitle(
            f'Presence heatmap for {species} with threshold of {threshold}',
            fontsize=10
            )
        ax = sns.heatmap(accumulated_presence.T, 
                    vmin=0,
                    # vmax=1,
                    cmap='viridis')
        locs, yticklabels = plt.yticks()
        if accumulate_by == 'day':
            labels = np.unique([ts.date() for ts in timestamps])
            selected_labels = labels[[int(i.get_text()) for i in yticklabels]]
        elif accumulate_by == 'month':
            labels = np.unique([ts.date() for ts in timestamps])
            selected_labels = labels[[int(i.get_text()) for i in yticklabels]]
        plt.yticks(locs, selected_labels)
        
        locs, labels = plt.xticks()
        x_idxs = [0, 6, 12, 18, 23]
        plt.xticks(locs[x_idxs], np.array(labels)[x_idxs])
        
        # plt.clabel('Binary presence each hour')
        
        fig.set_size_inches(6, 5)
        fig.set_dpi(300)
        fig.tight_layout()
        return fig      
    
    def accumulate_data(self, presence, timestamps, class_dict, species, accumulate_by='day'):
        species_idx = class_dict[species]
        species_presence = presence[:, species_idx]
        
        dates = np.array([getattr(ts, 'date')() for ts in timestamps])
        hours = np.array([getattr(ts, 'hour') for ts in timestamps])
        if accumulate_by == 'day':
            accumulated = self.transform_presence_into_hour_heatmap(
                            species_presence, hours, accumulator=dates
                        )
        elif accumulate_by == 'week':
            accumulated = np.zeros([24, len(np.unique(dates))], dtype=np.int8)
        elif accumulate_by == 'month':
            months = np.array([(getattr(ts, 'year'), getattr(ts, 'month')) for ts in timestamps])
            accumulated = self.transform_presence_into_hour_heatmap(
                species_presence, hours, accumulator=months
            )
        return accumulated
            
    @staticmethod
    def transform_presence_into_hour_heatmap(
        species_presence, hours, accumulator
        ):
        accumulated = np.zeros([24, len(np.unique(accumulator))], dtype=np.int8)
        for acc_idx, item in enumerate(np.unique(accumulator)):
            month_presence_idx = np.where(accumulator==item)[0]
            for hour in range(24):
                hourly_presence_idx = np.where(
                    hours[month_presence_idx]==hour
                    )[0]
                accumulated[hour, acc_idx] = sum(
                    species_presence[month_presence_idx[hourly_presence_idx]]
                    )
        return accumulated
        

    def get_timestamps_per_embedding(self, model):
        from bacpipe.embedding_evaluation.label_embeddings import get_dt_filename
        import datetime as dt
        
        embed_dict = self.vis_loader.embeds[model]
        ts_within_audio_files = [dt.timedelta(seconds=ts) for ts in embed_dict['timestamp']]
        ts_files = [get_dt_filename(f) for f in embed_dict['metadata']['audio_files']]
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
