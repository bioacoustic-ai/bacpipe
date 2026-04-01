import json

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.express as px

import bacpipe.embedding_evaluation.label_embeddings as le
from bacpipe import settings
# from bacpipe.embedding_evaluation.visualization.visualize_spectrograms import SpectrogramPlot
import matplotlib

import logging

logger = logging.getLogger(__name__)


COLOR_DISCRETE = px.colors.qualitative.Dark24


matplotlib.rcParams.update(
    {
        "figure.dpi": 600,  # High-resolution figures
        "savefig.dpi": 600,  # Exported plot DPI
        "font.size": 12,  # Better font readability
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def darken_hex_color_bitwise(hex_color):
    """
    Darkens a hex color using the bitwise operation: (color & 0xfefefe) >> 1.

    Parameters:
        hex_color (str): The hex color string (e.g., '#1f77b4').

    Returns:
        str: The darkened hex color.
    """
    # Remove '#' and convert hex color to an integer
    color_int = int(hex_color.lstrip("#"), 16)

    # Apply the bitwise operation to darken the color
    darkened_color_int = (color_int & 0xFEFEFE) >> 1

    # Convert back to a hex string and return with leading '#'
    return f"#{darkened_color_int:06x}"


def collect_dim_reduced_embeds(
    model_name, dim_reduced_embed_path, dim_reduction_model, **kwargs
):
    """
    Return the dimensionality reduced embeddings of a model.

    Parameters
    ----------
    model_name : str
        name of model
    dim_reduced_embed_path : pathlib.Path object
        path to dim reduced embeddings
    dim_reduction_model : str
        name of feature extraction model

    Returns
    -------
    dict
        dimensionality reduced embeddings
    """
    files = list(dim_reduced_embed_path.iterdir())
    if len(files) == 0:
        logger.warning(
            "No dimensionality reduced embeddings found for "
            f"{dim_reduction_model}. In fact the directory "
            f"{dim_reduced_embed_path} is empty. Deleting directory."
        )
        dim_reduced_embed_path.rmdir()
        dim_reduced_embed_path = le.get_dim_reduc_path_func(
            model_name, dim_reduction_model=dim_reduction_model, **kwargs
        )
        files = list(dim_reduced_embed_path.iterdir())
    for file in files:
        if file.suffix == ".json":  # and dim_reduction_model in file.stem:
            with open(file, "r") as f:
                embeds_dict = json.load(f)
    return embeds_dict


class EmbedAndLabelLoader:
    def __init__(self, dim_reduction_model, dashboard=False, **kwargs):
        self.labels = dict()
        self.embeds = dict()
        self.split_data = dict()
        self.bool_noise = dict()
        self.dashboard = dashboard
        self.dim_reduction_model = dim_reduction_model
        self.kwargs = kwargs

    def get_data(self, model_name, label_by, remove_noise=False, **kwargs):
        if not model_name in self.labels.keys():

            tup = get_labels_for_plot(model_name, **self.kwargs)
            self.labels[model_name], self.bool_noise[model_name] = tup

            dim_reduced_embed_path = le.get_dim_reduc_path_func(
                model_name, dim_reduction_model=self.dim_reduction_model, **kwargs
            )

            self.embeds[model_name] = collect_dim_reduced_embeds(
                model_name, dim_reduced_embed_path, self.dim_reduction_model, **kwargs
            )

        if remove_noise:
            return_embeds, return_labels = self.remove_noise_indices(model_name)
        else:
            return_labels = self.labels[model_name]
            return_embeds = self.embeds[model_name]
            return_embeds['index'] = np.arange(len(return_embeds['x']))
            if (
                len(return_embeds['metadata']['audio_files'])
                < len(return_embeds['x'])
                ):
                audiofilenames = []
                [
                    audiofilenames.extend([f] * nr) 
                    for f, nr in zip(
                        return_embeds['metadata']['audio_files'], 
                        return_embeds['metadata']['nr_embeds_per_file']
                        )
                ]
                return_embeds['metadata']['audio_files'] = audiofilenames

        if label_by in return_labels:
            return_splits = data_split_by_labels(return_embeds, return_labels[label_by])
        else:
            return [], [], {}
        return (
            return_labels[label_by],
            return_embeds,
            return_splits,
        )

    def remove_noise_indices(self, model_name):
        return_labels, return_embeds = dict(), dict()
        bool_noise = self.bool_noise[model_name]
        
        
        for key, values in self.labels[model_name].items():
            if "noise" in key:
                return_labels[key] = values
            else:
                return_labels[key] = np.array(values, 
                                                dtype=object)[~bool_noise]

        for key, value in self.embeds[model_name].items():
            if not key == 'metadata':
                return_embeds[key] = np.array(value)[~bool_noise]
            else:
                return_embeds['metadata'] = dict()
                for meta_key, meta_value in value.items():
                    if not isinstance(meta_value, list):
                        return_embeds['metadata'][meta_key] = meta_value
                    else:
                        if meta_key == 'audio_files':
                            return_embeds['metadata'][meta_key] = (
                                np.array(meta_value)[~bool_noise]
                                )
        return return_embeds, return_labels

def plot_embeddings(
    loader,
    model_name,
    label_by,
    paths=None,
    dim_reduction_model=None,
    axes=False,
    fig=False,
    dashboard=False,
    dashboard_idx=None,
    **kwargs,
):
    """
    Generate figures and axes to plot points corresponding to embeddings.
    This function can also be called and given figure and axes handeles.
    In that case the existing handles will be used to add the points and
    configure the axes and labels.

    Parameters
    ----------
    loader : EmbedAndLabelLoader object
        contains the labels and embeddings by model, for quicker loading
    model_name : str
        name of model
    label_by : str, optional
        key of default_labels dict, by default "audio_file_name"
    paths : SimpleNamespace object, optional
        object with path attributes, defaults to None
    dim_reduction_model : str
        name of dim reduced model
    axes : plt object, optional
        axes handle, by default False
    fig : plt object, optional
        figure handle, by default False
    dashboard : bool, optional
        whether the calls comes from the dashboard, by deafult False
    dashboard_idx : int, optional
        index of dashboard plot, relevant for legend placement

    Returns
    -------
    plt object
        axes handles is axes handles were given
    dict
        color dictionary for legend
    list
        plt point objects for legend of colorbar
    """
    labels, embeds, split_data = loader.get_data(model_name, label_by, **kwargs)

    fig, axes, return_axes = init_embed_figure(fig, axes, **kwargs)
    
    if len(labels) == 0 and len(embeds) == 0:
        return fig

    if label_by == 'audio_file_name':
        new_labels = [Path(l).stem+Path(l).suffix for l in labels]
        new_split_data = dict()
        for label in split_data.keys():
            new_label = Path(label).stem+Path(label).suffix
            new_split_data[new_label] = split_data[label]
        split_data = new_split_data

    c_label_dict = {lab: i for i, lab in enumerate(np.unique(labels))}
    points = plot_embedding_points(
        axes, embeds, split_data, labels, c_label_dict, **kwargs
    )

    if return_axes:
        return axes, c_label_dict, points
    elif dashboard:
        if True:
            # return plotly_mutual_information(
            return plot_embeddings_px(
                embeds, 
                labels,
                c_label_dict,
                label_by=label_by
            )
        else:
            fig.set_size_inches(6, 5)
            fig.set_dpi(300)
            fig.tight_layout()
            set_colorbar_or_legend(
                fig,
                axes,
                points,
                c_label_dict,
                dashboard=dashboard,
                label_by=label_by,
                **kwargs,
            )
            return fig
    else:
        set_colorbar_or_legend(
            fig, axes, points, c_label_dict, label_by=label_by, **kwargs
        )

        axes.set_title(f"{dim_reduction_model.upper()} embeddings")
        fig.savefig(paths.plot_path.joinpath("embeddings.png"), dpi=300)
        plt.close(fig)

def init_embed_figure(fig, axes, bool_3d=False, **kwargs):
    if not fig:
        if bool_3d:
            fig, axes = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(figsize=(12, 8), dpi=400)
        return_axes = False
    else:
        return_axes = True
    axes.set_xticks([])
    axes.set_yticks([])
    return fig, axes, return_axes


def get_labels_for_plot(model_name=None, **kwargs):
    labels = dict()
    labels = le.get_default_labels(model_name, **kwargs)

    if le.get_paths(model_name).labels_path.joinpath("ground_truth.npy").exists():
        ground_truth = le.get_ground_truth(model_name)
        for label_column in [key for key in ground_truth.keys() if "label:" in key]:
            label = label_column.split("label:")[-1]
            inv = {v: k for k, v in ground_truth[f"label_dict:{label}"].items()}
            inv[-1.0] = "noise"
            inv[-2.0] = "noise"
            # technically -2.0 is not noise, but corresponds to sections
            # with multiple sources vocalizing simultaneously
            if len(ground_truth[label_column].shape) > 1:
                # TODO for display we're just taking the first label
                labels[label] = [inv[v] for v in ground_truth[label_column][:, 0]]
            else:
                labels[label] = [inv[v] for v in ground_truth[label_column]]
            bool_noise = np.array(labels[label]) == "noise"
    else:
        bool_noise = np.array([False] * len(list(labels.values())[0]))
    if len(list(le.get_paths(model_name).clust_path.glob("*.npy"))) > 0:
        clusts = [
            np.load(f, allow_pickle=True).item()
            for f in le.get_paths(model_name).clust_path.glob("*.npy")
        ]
        for clust in clusts:
            for name, values in clust.items():
                if "kmeans" in name:
                    labels[name] = values
                else:
                    labels[name] = np.array(["noise"] * len(bool_noise), dtype=object)
                    labels[name][~bool_noise] = [inv[v] for v in values]

    return labels, bool_noise


def set_colorbar_or_legend(fig, axes, points, c_label_dict, label_by, **kwargs):
    if len(c_label_dict.keys()) > 20:
        if isinstance(list(c_label_dict.keys())[0], int):
            fontsize = 9
        elif isinstance(list(c_label_dict.keys())[0], np.int32):
            fontsize = 9
        elif len(list(c_label_dict.keys())[0]) < 12:
            fontsize = 9
        else:
            fontsize = 6

        # Shrink main plot area to make space for colorbar
        fig.subplots_adjust(right=0.7)

        # Add colorbar axis manually (x0, y0, width, height) in figure coords
        cbar_ax = fig.add_axes([0.72, 0.05, 0.03, 0.9])  # tweak as needed

        # Create colorbar in the custom axis
        cbar = fig.colorbar(points, cax=cbar_ax)

        locs = [*(int(len(c_label_dict) / 5) * np.arange(5)), -1]
        cbar.set_ticks([list(c_label_dict.values())[loc] for loc in locs])
        cbar.set_ticklabels(
            [list(c_label_dict.keys())[loc] for loc in locs], fontsize=fontsize
        )
        cbar.set_label(label_by.replace("_", " "), fontsize=10)
    else:
        hands, labs = axes.get_legend_handles_labels()
        fig, axes = set_legend(hands, labs, fig, axes, **kwargs)
    return fig, axes


def plot_embedding_points(
    axes, embeds, split_data, labels, c_label_dict, remove_noise=False, **kwargs
):
    """
    Plot embeddings in scatter plot.

    Parameters
    ----------
    axes : plt object
        axes handle
    embeds : dict
        embeddings
    split_data : dict
        data split by label
    labels : list
        labels of the data
    c_label_dict : dict
        linking labels to ints for coloring
    remove_noise : bool, optional
        remove noise or not, defaults to False

    Returns
    -------
    plt object
        axes points
    """
    if len(c_label_dict.keys()) > 20:
        import matplotlib.cm as cm

        cmap = cm.viridis

        num_labels = np.array([c_label_dict[lab] for lab in labels])
        if not len(labels) == len(embeds['x']):
            raise AssertionError(
                f"The number of labels is {len(labels)} whereas the number of "
                f"embedding points is {len(embeds['x'])}. This mismatch could "
                "be the result of an incomplete run and bacpipe is using "
                "the dim_reduced_embeddings corresponding to that. Check if in your results folder "
                "there are not multiple dim_reduced_embeddings, and if so, delete the incomplete one."
            )
        if len(np.array(embeds['x']).shape) > 1:
            embeds["x"] = np.array(embeds["x"])[:, 0],
            embeds["y"] = np.array(embeds["y"])[:, 0],
        points = axes.scatter(
            np.array(embeds["x"]),
            np.array(embeds["y"]),
            c=num_labels,
            label=labels,
            s=1,
            cmap=cmap,
        )
    else:
        cmap = plt.cm.tab20
        colors = cmap(np.arange(len(c_label_dict.keys())) % cmap.N)
        for idx, (label, data) in enumerate(split_data.items()):
            if remove_noise and label == "noise":
                continue
            points = axes.scatter(
                data[0],
                data[1],
                label=label,
                s=1,
                color=colors[idx],
            )
    return points


def set_legend(
    handles, labels, fig, axes, bool_plot_centroids=False, dashboard=False, **kwargs
):
    """
    Create the legend for embeddings visualization plots.

    Parameters
    ----------
    handles : list
        list of legend handles
    labels : list
        list of labels for legend
    fig : plt.fig object
        figure handle
    axes : plt.axes object
        axes handle
    bool_plot_centroids : bool, optional
        if True centroids of each class will be plotted, by default True
    dashboard : bool
        if dashboard called this function or not

    Returns
    -------
    plt.fig object
        figure handle
    plt.axes object
        axes handle
    """

    # Calculate number of columns dynamically based on the number of labels
    num_labels = len(labels)  # Number of labels in the legend
    ncol = min(num_labels, 5)  # Use 6 columns or fewer if there are fewer labels

    if bool_plot_centroids:
        custom_marker = plt.scatter(
            [], [], marker="x", color="black", s=10
        )  # Empty scatter, only for the legend
        new_handles = handles[::2] + [custom_marker]
        new_labels = labels[::2] + ["centroids"]
    else:
        new_handles = handles
        new_labels = labels
    if dashboard:
        fig.subplots_adjust(right=0.7)

        fig.legend(
            new_handles,
            new_labels,
            loc="outside right",
            markerscale=4 if dashboard else 6,
            fontsize=7,
            frameon=False,
        )
    else:

        fig.subplots_adjust(bottom=0.2)
        fig.legend(
            new_handles,
            new_labels,  # Use the handles and labels from the plot
            loc="outside lower center",  # Center the legend
            ncol=ncol,  # Number of columns
            markerscale=6,
        )
    return fig, axes


def data_split_by_labels(embeds_dict, labels):
    """
    Split data by labels for scatterplots.

    Parameters
    ----------
    embeds_dict : dict
        embeddings by model
    labels : list
        list of labels

    Returns
    -------
    dict
        x and y data corresponding to labels
    """
    split_data = {}
    uni_labels = np.unique(labels)
    if len(uni_labels) > 20:
        split_data["all"] = np.array(
            [
                np.array(embeds_dict["x"]),
                np.array(embeds_dict["y"]),
            ]
        )
    else:
        for label in uni_labels:  # TODO don't do this for more than 20 categories
            split_data[str(label)] = np.array(
                [
                    np.array(embeds_dict["x"])[np.array(labels) == label],
                    np.array(embeds_dict["y"])[np.array(labels) == label],
                ]
            )

    return split_data


def return_rows_cols(num):
    if num <= 3:
        return 1, 3
    elif num > 3 and num <= 6:
        return 2, 3
    elif num > 6 and num <= 9:
        return 3, 3
    elif num > 9 and num <= 12:
        return 3, 4
    elif num > 12 and num <= 16:
        return 4, 4
    elif num > 16 and num <= 20:
        return 4, 5
    else:
        return 5, num // 5


def set_figsize_for_comparison(rows, cols):
    if rows == 1:
        return (11, 5)
    elif rows == 2:
        return (11, 7)
    elif rows == 3:
        return (11, 8)
    elif rows > 3:
        return (11, 10)


def plot_comparison(
    plot_path,
    models,
    dim_reduction_model,
    bool_spherical=False,
    dashboard=False,
    loader=None,
    evaluation_task=[],
    **kwargs,
):
    """
    Create big overview visualization of all embeddings spaces. Labels
    are chosen from ground_truth and if that does not exist, default
    lables are used.

    Parameters
    ----------
    plot_path : pathlib.Path object
        path to store overview plots
    models : list
        list of models
    dim_reduction_model : str
        name of dimensionality reduction model
    bool_spherical : bool, optional
        if True 3d embeddings will be plotted, by default False
    dashboard : bool, optional
        if dashboard called this function or not
    loader : EmbedAndLabelLoader object
        object containing embeds and labels by model for quicker loading
    evaluation_task : list, optional
        list of tasks to evaluate, by default []

    Returns
    -------
    plt object
        figure handle
    """
    rows, cols = return_rows_cols(len(models))

    if not bool_spherical:
        fig = Figure(figsize=set_figsize_for_comparison(rows, cols))
        axes = fig.subplots(rows, cols)
    else:
        fig = Figure(figsize=set_figsize_for_comparison(rows, cols))
        axes = fig.subplots(
            rows,
            cols,
            subplot_kw={"projection": "3d"},
        )
    if not dashboard:
        vis_loader = EmbedAndLabelLoader(dim_reduction_model, **kwargs)
    else:
        vis_loader = loader

    c_label_dict, points = {}, {}
    for idx, model in enumerate(models):
        paths = le.get_paths(model)

        axes.flatten()[idx], c_label_dict[idx], points[idx] = plot_embeddings(
            vis_loader,
            model,
            paths=paths,
            dim_reduction_model=dim_reduction_model,
            axes=axes.flatten()[idx],
            fig=fig,
            bool_plot_centroids=False,
            dashboard=dashboard,
            **kwargs,
        )
        axes.flatten()[idx].set_title(f"{model.upper()}")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.2)
    colorbar_idx = np.argmax([len(d) for d in c_label_dict.values()])

    fig, _ = set_colorbar_or_legend(
        fig,
        axes.flatten()[colorbar_idx],
        points[colorbar_idx],
        c_label_dict[colorbar_idx],
        dashboard=dashboard,
        **kwargs,
    )
    [ax.remove() for ax in axes.flatten()[idx + 1 :]]
    if "clustering" in evaluation_task:
        reorder_embeddings_by_clustering_performance(plot_path, axes, models)

    fig.suptitle(f"Comparison of {dim_reduction_model} embeddings", fontweight="bold")
    if not dashboard:
        fig.savefig(plot_path.joinpath("comp_fig.png"), dpi=300)
        plt.close(fig)
    else:
        return fig


def reorder_embeddings_by_clustering_performance(
    plot_path, axes, models, order_metric="ground_truth-kmeans"
):
    """
    Reorder the embedding overview plot by clustering performance.

    Parameters
    ----------
    plot_path : pathlib.Path object
        path to store plots and results comparing all models
    axes : plt.axes object
        handle for figures axes
    models : list
        list of models
    order_metric : str
        key corresponding to a metric in the clustering_results.json file.
        Defaults to "ARI(kmeans)"
    """
    clust_dict = json.load(open(plot_path.joinpath("clustering_results.json"), "r"))
    new_order = dict(
        sorted(
            clust_dict.items(), key=lambda kv: kv[1]["ARI"][order_metric], reverse=True
        )
    )
    positions = {mod: ax.get_position() for mod, ax in zip(new_order, axes.flatten())}
    for model, ax in zip(models, axes.flatten()):
        if not model in positions.keys():
            continue
        ax.set_position(positions[model])





def plot_embeddings_px(
    embeds,
    labels,
    c_label_dict,
    label_by="label",
    **kwargs
):
    # 1. Prepare Data
    if len(np.array(embeds['x']).shape) > 1:
        embeds['x'] = np.array(embeds['x']).squeeze()
        embeds['y'] = np.array(embeds['y']).squeeze()
    x_data = embeds['x']
    y_data = embeds['y']
    
    audiofilenames = embeds['metadata']['audio_files']
    
    starts = embeds['timestamp']
    ends = np.array(starts) + (
        embeds['metadata']['segment_length (samples)'] 
        / embeds['metadata']['sample_rate (Hz)']
        )
    ends = ends.tolist()
    
    # Calculate unique labels to decide on Legend vs Colorbar
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Create an integer mapping for high-cardinality plotting
    # (Plotly needs numbers to generate a gradient colorbar)
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    label_ids = [label_to_id[l] for l in labels]

    df = pd.DataFrame({
        'x': x_data,
        'y': y_data,
        'label': labels,            # The actual string (for hover/legend)
        'label_id': label_ids,      # The integer (for colorbar)
        'audiofilename': audiofilenames,
        'start': starts,
        'end': ends,
        'idx': embeds['index']
    })

    # 2. Setup Figure based on Label Count
    if n_labels > 50:
    # if label_by in ['time_of_day', 'continuous_timestamp', 'day_of_year']:
        # --- HIGH CARDINALITY: Use Colorbar ---
        # We map color to 'label_id' (int) to force a continuous scale
        fig = px.scatter(
            df, x='x', y='y',
            color='label_id',
            hover_data={
                'x': False,
                'y': False,
                'label': True,
                'label_id': False,  
                'audiofilename': True, 
                'start':True, 
                'end':True
                },
            custom_data=['audiofilename', 'start', 'end', 'idx'],
            title=f"Embedding Plot - {embeds['metadata']['model_name']} - {label_by}",
            render_mode='webgl',
            color_continuous_scale=kwargs.get('color_continuous')
        )

        tick_vals = np.linspace(0, n_labels - int(n_labels//100+1), 6).astype(int).tolist()
        tick_text = [str(unique_labels[i]) for i in tick_vals]
        
        fig.update_coloraxes(
                    colorbar_title=label_by,
                    colorbar_tickmode='array',
                    colorbar_tickvals=tick_vals,
                    colorbar_ticktext=tick_text,
                )


    else:
        
        # force a discrete legend
        fig = px.scatter(
            df, x='x', y='y',
            color='label', 
            hover_data={
                'x': False,
                'y': False,
                'label': True,
                'label_id': False,  
                'audiofilename': True, 
                'start':True, 
                'end':True
                },
            custom_data=['audiofilename', 'start', 'end', 'idx'],
            title=f"Embedding Plot - {embeds['metadata']['model_name']} - {label_by}",
            render_mode='webgl',
            color_discrete_sequence=COLOR_DISCRETE
        )
        
        # Configure the Discrete Legend
        fig.update_layout(
            legend=dict(
                orientation="v", 
                yanchor="bottom", 
                y=0, 
                xanchor="left", 
                x=1.02,
                title_text=label_by
            )
        )

    
    fig.update_layout(
        # autosize=True,
        template='plotly_white',
        height=settings.embed_fig_height,    
        clickmode='event', 
        hovermode='closest',
        # margin=dict(l=20, r=20, t=40, b=20),
        margin=dict(l=0, r=80, t=40, b=0),
        # Ensure selection tools are available
        modebar=dict(add=['lasso2d', 'select2d'], remove=['autoScale2d']),
    )
    # fig.update_xaxes(visible=False, showticklabels=True) # Hide x axis ticks 
    # fig.update_yaxes(visible=False, showticklabels=True) # Hide y axis ticks
    
    # Improve marker appearance
    fig.update_traces(marker_size=8, marker_opacity=0.6)
    return fig

