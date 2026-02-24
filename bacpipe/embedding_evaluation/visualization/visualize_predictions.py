import json

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import plotly.express as px
import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from bacpipe.embedding_evaluation.classification.train_classifier import (
    LinearClassifier
    )




def plot_classification_results(
    task_name,
    paths=None,
    metrics=None,
    return_fig=False,
    path_func=None,
    model_name=None,
):
    """
    Save model specific classification results in the model specific
    plot path, displayed as horizontal bars.

    Parameters
    ----------
    task_name : str
        name of task
    paths : SimpleNamespace object
        path to store plots
    metrics : dict
        classification performance
    return_fig : bool
        if True the figure will be returned, by default False
    path_func : function
        function to return the paths when model name is given
    model_name : str
        name of model, by default None

    Returns
    -------
    plt object
        figure handle
    """
    if path_func and model_name:
        paths = path_func(model_name)
    if not metrics:
        class_path = paths.class_path / f"class_results_{task_name}.json"
        if not class_path.exists():
            error = (
                f"\nThe classification file {class_path} does not exist. Perhaps it was not "
                "created yet. To avoid getting this error, make sure you have not "
                " included 'classification' in the 'evaluation_tasks'. If you want to compute "
                "classification, make sure to set `overwrite=True`."
            )
            logger.exception(error)
            raise AssertionError(error)

        with open(paths.class_path / f"class_results_{task_name}.json", "r") as f:
            metrics = json.load(f)

    # Filter overall metrics if needed
    metrics["overall"] = {
        k: v for k, v in metrics["overall"].items() if not "micro" in k
    }

    # Sort classes by accuracy for better visualization
    class_items = sorted(
        metrics["per_class_accuracy"].items(), key=lambda x: x[1], reverse=True
    )
    class_names = [item[0] for item in class_items]
    class_values = [item[1] for item in class_items]

    # Set figure size based on number of classes and return_fig
    if return_fig:
        # For dashboard, make height adapt to number of classes
        height = max(4, len(class_names) * 0.3)
        fig, ax = plt.subplots(1, 1, figsize=(5, height))
        fontsize = 10
    else:
        height = max(8, len(class_names) * 0.4)
        fig, ax = plt.subplots(1, 1, figsize=(12, height))
        fontsize = 14

    model_name = paths.labels_path.parent.stem
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(class_names)) % cmap.N)

    # Create horizontal bars
    ax.barh(
        range(len(class_names)),
        class_values,
        height=0.6,
        color=colors,
    )

    # Create metrics string
    metrics_string = "".join(
        [f"{k}: {v:.3f} | " for k, v in metrics["overall"].items()]
    )

    fig.suptitle(
        f"Classwise accuracy for {task_name} "
        f"classification with {model_name.upper()} embeddings\n"
        f"{metrics_string}",
        fontsize=fontsize,
    )

    # Adjust labels for horizontal orientation
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Classes")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)

    # Add value labels at the end of each bar
    for i, v in enumerate(class_values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    # Set x-axis limits for better visualization
    ax.set_xlim(0, min(1.0, max(class_values) * 1.15))

    # Add grid lines for easier reading
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if return_fig:
        return fig

    path = paths.plot_path
    fig.savefig(
        path.joinpath(f"class_results_{task_name}_{model_name}.png"),
        dpi=300,
    )
    plt.close(fig)


def load_results(path_func, task, model_list):
    """
    Load the task results into a dict and return them. For classification
    multiple subtasks exist, so do them seperately.

    Parameters
    ----------
    path_func : function
        returns model specific tasks when model is given
    task : str
        name of task
    model_list : list
        list of models

    Returns
    -------
    dict
        performance for different tasks and models
    """
    metrics = {}
    for model_name in model_list:
        paths = path_func(model_name)
        for file in getattr(paths, f"{task[:5]}_path").rglob("*results*.json"):
            if task == "classification":
                subtask = file.stem.split("_")[-1]
                metrics[f"{model_name}({subtask})"] = json.load(open(file, "r"))
            else:
                metrics[model_name] = json.load(open(file, "r"))
    return metrics



def plot_per_class_metrics(plot_path, task_name, model_list, metrics):
    """
    Visualization of per class results. Resulting figure is stored in
    plot path. Models are sorted by the value of the first entry.

    Parameters
    ----------
    plot_path : pathlib.Path object
        path to store plot in
    task_name : str
        name of task
    model_list : list
        list of models
    metrics : dict
        performance dictionary
    """
    per_class_metrics = {m: v["per_class_accuracy"] for m, v in metrics.items()}
    overall_metrics = {m: v["overall"] for m, v in metrics.items()}
    num_classes = len(per_class_metrics[model_list[0]].keys())
    fig_width = max(12, num_classes * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

    cmap = plt.cm.tab10
    model_colors = cmap(np.arange(len(model_list)) % cmap.N)

    d = {m: v["macro_accuracy"] for m, v in overall_metrics.items()}
    model_list = sorted(d, key=d.get, reverse=True)
    all_classes = sorted(per_class_metrics[model_list[0]].keys())

    for i, model_name in enumerate(model_list):
        class_values = per_class_metrics[model_name].values()

        ax.scatter(
            np.arange(len(class_values)),
            class_values,
            color=model_colors[i],
            label=f"{model_name.upper()} "
            + f"(accuracy: {overall_metrics[model_name]['macro_accuracy']:.3f})",
            s=100,
        )

        ax.plot(
            np.arange(len(class_values)),
            class_values,
            color=model_colors[i],
            linestyle="-",  # Solid line
            linewidth=1.5,
        )

    fig.suptitle(
        f"Per class metrics for {task_name} across models",
        fontsize=14,
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Classes")
    ax.set_xticks(np.arange(len(all_classes)))
    ax.set_xticklabels(all_classes, rotation=90)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Models", fontsize=10)

    fig.subplots_adjust(right=0.65, bottom=0.3)
    file_name = (
        f"comparison_{task_name.replace(' ', '_')}_" 
        + "-".join([m[:2] for m in model_list]) 
        + ".png"
    )
    plot_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(
        plot_path.joinpath(file_name),
        dpi=300,
    )
    plt.close(fig)

import plotly.express as px
import numpy as np
from bacpipe.embedding_evaluation.visualization.visualize_spectrograms import SpectrogramPlot

def plot_classification_heatmap(
    event, predictions_loader, model, accumulate_by, 
    threshold, species=None, **kwargs
    ):
    if event is None and species is None:
        return SpectrogramPlot.dummy_image(title="Click the button to generate a prediction heatmap.")
    try:
        predictions_loader.get_data(model, threshold, **kwargs)
    except Exception as e:
        logger.exception(e)
        # TODO thid doesn't update the plot for some reason
        return SpectrogramPlot.dummy_image(title=str(e)) 
    # if predictions_loader.binary_presence is None:
    #     return predictions_loader.failed_fig
    accumulated_presence = predictions_loader.accumulate_data(species, accumulate_by)
    timestamps = predictions_loader.timestamps
    
    logger.info('Redrawing heatmap plot')
    
    # Prepare data - mask values below 0
    plot_data = accumulated_presence.T.copy()
    plot_data = np.where(plot_data < 0, np.nan, plot_data)
    
    # Get time labels based on accumulation type
    if accumulate_by == 'day':
        y_labels = [str(ts.date()) for ts in timestamps]
        y_axis_label = 'Dates'
    elif accumulate_by == 'month':
        y_labels = [f'{date.year}-{date.month}' for date in timestamps]
        y_axis_label = 'Months'
    elif accumulate_by == 'week':
        y_labels = [f'{date.year}-W{date.isocalendar().week}' for date in timestamps]
        y_axis_label = 'Weeks'
    
    # Hour labels (0-23)
    x_labels = list(range(24))
    
    # Create heatmap
    fig = px.imshow(
        plot_data,
        labels=dict(x="Hours", y=y_axis_label, color="Binary presence per hour"),
        x=x_labels,
        y=np.unique(y_labels),
        color_continuous_scale='Viridis',
        zmin=0,  # Values below this will be white (nan handling)
        aspect="auto",
        title=(
            f'Presence heatmap using {model} with '
            f'{predictions_loader.current_clfier_type} classifier <br>'
            f'for {species} '
            f'with threshold of {PredictionsLoader.verify_threshold(threshold)}.'
            )
    )
    
    # Customize layout
    fig.update_layout(
        autosize=True,
        # width=600,
        height=500,
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 6, 12, 18, 23],
            ticktext=['0', '6', '12', '18', '23']
        ),
        yaxis=dict(
            autorange='reversed'  # Optional: match seaborn orientation
        ),
        coloraxis_colorbar=dict(
            title="Binary presence per hour"
        )
    )
    
    # Make NaN values appear white
    fig.update_traces(
        hovertemplate='Hour: %{x}<br>' + y_axis_label + ': %{y}<br>Presence: %{z}<extra></extra>'
    )
    
    return fig


class PredictionsLoader:
    def __init__(
        self, vis_loader, path_func, models, 
        panel_selection, progress_bar, loading_pane, 
        thresh=0.5
        ):
        self.vis_loader = vis_loader
        self.path_func = path_func
        self.models = models
        self.thresh = thresh
        self.panel_selection = panel_selection
        self.progress_bar = progress_bar
        self.loading_pane = loading_pane
    
    def get_data(
        self, model, threshold, 
        clfier_type=None, clfier_path='', **kwargs
        ):
        threshold = self.verify_threshold(threshold)
        if hasattr(self, 'binary_presence'):
            if (
                self.current_model == model
                and self.current_threshold == threshold
                and (
                    self.current_clfier_type == clfier_type
                    or clfier_type is None
                    )
                ):
                return
            
        self.current_model = model
        self.current_threshold = threshold
        self.current_clfier_type = clfier_type
            
        
        if not (
            self.path_func(self.models[0]).class_path / 'linear_classifier.pt'
            ).exists():
            clfier_type = 'Integrated'
        
        if clfier_type == 'Linear':
            self.loading_pane.value = 'Loading embeddings for classification'
            self.binary_presence, self.class_dict = self.classify_embeddings(
                model, threshold, clfier_path
            )
        elif clfier_type == 'Integrated':
            self.loading_pane.name = 'Preparing heatmap'
            self.loading_pane.value = 'Loading precomputed embeddings'
            self.binary_presence, self.class_dict = self.load_classification(
                model, threshold
                )
        
        
        self.embed_dict = self.vis_loader.embeds[model]
        
        if self.binary_presence is None:
            raise FileNotFoundError(
                "It seems like the classifier hasn't been run yet, or "
                f"that {model} doesn't have a pretrained classifier. "
                "If the model has a pretrained classifier, please rerun "
                "bacpipe with the setting `run default classifier` set to `True`."
            )
        
        self.get_timestamps_per_embedding(model)
        
        self.class_dict['overall'] = len(self.class_dict)
        self.binary_presence = np.concatenate(
            [self.binary_presence.T, 
             [np.sum(self.binary_presence, axis=1).astype(np.int8)]]
            ).T
        
        
        self.class_dict = self.reorder_by_most_occurrance(
            self.binary_presence, self.class_dict
            )
        
        self.panel_selection.options = list(self.class_dict.keys())        
    
    
    def collect_all_embeddings(self, model, Loader, config, settings):
        ld = Loader(
            audio_dir=config.audio_dir, 
            model_name=model,
            **vars(settings)
            )
        
        embeds = ld.embeddings(as_type='array')
        return embeds
    
    def run_classifier(self, embeds, linear_clfier, threshold):
        import torch.nn.functional as F
        probs = []
        for idx, batch in enumerate(embeds):
            logits = linear_clfier(batch)
            probabilities = F.softmax(logits, dim=0).detach().cpu().numpy()
            binary_classification = np.zeros(probabilities.shape, dtype=np.int8)
            binary_classification[probabilities > threshold] = 1
            probs.append(binary_classification.tolist())
            self.progress_bar.value = int((idx+1)/len(embeds)*100)
        return np.array(probs, dtype=np.int8)
            
    @staticmethod
    def verify_threshold(threshold):
        if threshold == '':
            threshold = 0.5
        else:
            threshold = float(threshold)
        return threshold
        
    
    def classify_embeddings(self, model, threshold, clfier_path):
        import torch
        from bacpipe.core.experiment_manager import Loader
        from bacpipe import config, settings
        if clfier_path == '':
            clfier_path = (
                self.path_func(model).class_path / 'linear_classifier.pt'
                ).as_posix()
        threshold = self.verify_threshold(threshold)
        
        embeds = self.collect_all_embeddings(model, Loader, config, settings)
        
        embeds = torch.Tensor(embeds).to(settings.device)
        
        self.loading_pane.value = 'Running classifier'
        
        with open(Path(clfier_path).parent / 'label2index.json', 'r') as f:
            label2index = json.load(f)
            
        clfier_weights = torch.load(clfier_path, map_location=settings.device)
        clfier = LinearClassifier(
            clfier_weights['clfier.weight'].shape[-1], 
            len(label2index)
            )
        clfier.load_state_dict(clfier_weights)
        clfier.to(settings.device)
        
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
        
    def get_classes(self, path):
        if path == '':
            path = (
                self.path_func(self.models[0]).class_path / 'linear_classifier.pt'
                )
        if path.exists():
            with open(Path(path).parent / 'label2index.json', 'r') as f:
                classes = json.load(f)
            return list(classes.keys())
        else:
            return []
            
    def load_classification(self, model, threshold):
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
        keys2idx = {}
        for idx, file in enumerate(files):
            with open(file, 'r') as f:
                d = json.load(f)
                current_time_bins = d['head']['Time bins in this file']
                d.pop('head')
                
                for k, v in d.items():
                    cl_dict[k] = np.zeros([total_length + current_time_bins])    
                    if not keys2idx:
                        keys2idx[k] = 0
                    if not k in keys2idx:
                        keys2idx[k] = max(keys2idx.values()) + 1
                        
                    cl_dict[k][np.array(v['time_bins_exceeding_threshold']) + total_length] = v['classifier_predictions']
                    # file_specific_classification[v['time_bins_exceeding_threshold'], k2idx[k]] = v['classifier_predictions']
                for species in [
                    k for k, v in cl_dict.items() 
                    if len(v) < total_length + current_time_bins
                    ]:
                    cl_dict[species] = np.hstack([cl_dict[species], np.zeros([current_time_bins])])
                
                total_length += current_time_bins
            self.progress_bar.value = int((idx+1)/len(files)*100)
        
        probs_array = np.array(list(cl_dict.values())).T
        # binary_classification = probs_array[probs_array > thresh]
        binary_classification = np.zeros(probs_array.shape, dtype=np.int8)
        binary_classification[probs_array > threshold] = 1
        
        return binary_classification, keys2idx
    
    def accumulate_data(
        self, species, accumulate_by='day'
        ):
        if not species:
            species = 'overall'
        self.panel_selection.value = species
        species_idx = self.class_dict[species]
        species_presence = self.binary_presence[:, species_idx]
        
        dates = np.array([ts.date() for ts in self.timestamps])
        hours = np.array([ts.hour for ts in self.timestamps])
        if accumulate_by == 'day':
            date_tuple = [(d.year, d.month, d.day) for d in dates]
            accumulated = self.transform_presence_into_hour_heatmap(
                species_presence, hours, accumulator=date_tuple
                )
        elif accumulate_by == 'week':
            week_tuple = [(date.year, date.isocalendar().week) for date in dates]
            accumulated = self.transform_presence_into_hour_heatmap(
                species_presence, hours, accumulator=week_tuple
                )
        elif accumulate_by == 'month':
            month_tuple = [(date.year, date.month) for date in dates]
            accumulated = self.transform_presence_into_hour_heatmap(
                species_presence, hours, accumulator=month_tuple
                )
        return accumulated
            
    @staticmethod
    def transform_presence_into_hour_heatmap(
        species_presence, hours, accumulator
        ):
        accumulated = np.ones(
            [24, len(np.unique(accumulator, axis=0))]
            , dtype=np.int8
            ) *-1
        for acc_idx, item in enumerate(np.unique(accumulator, axis=0)):
            month_presence_idx = np.where(np.all(accumulator == item, axis=1))[0]
            for hour in range(24):
                hourly_presence_idx = np.where(
                    hours[month_presence_idx]==hour
                    )[0]
                if len(hourly_presence_idx) > 0:
                    accumulated[hour, acc_idx] = sum(
                        species_presence[month_presence_idx[hourly_presence_idx]]
                        )
        return accumulated
        
    def get_timestamps_per_embedding(self, model):
        from bacpipe.embedding_evaluation.label_embeddings import get_dt_filename
        import datetime as dt
        
        # embed_dict = self.vis_loader.embeds[model]
        ts_within_audio_files = [dt.timedelta(seconds=ts) for ts in self.embed_dict['timestamp']]
        ts_files = [get_dt_filename(f) for f in self.embed_dict['metadata']['audio_files']]
        ts_files_same_length_as_embeds = []
        [
            ts_files_same_length_as_embeds.extend([ts_file] * embed_len) 
            for ts_file, embed_len in zip(ts_files, self.embed_dict['metadata']['nr_embeds_per_file'])
        ]
        
        self.timestamps = [
            ts_file+ts_within_audio_file
            for ts_file, ts_within_audio_file in 
            zip(ts_files_same_length_as_embeds, ts_within_audio_files)
            ]
        