import json

import matplotlib.pyplot as plt

plt.ioff()
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

    Parameters
    ----------
    hex_color : str
        The hex color string (e.g., '#1f77b4').

    Returns
    -------
    str
        The darkened hex color.
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
            "\nNo dimensionality reduced embeddings found for "
            f"{dim_reduction_model}. In fact the directory "
            f"{dim_reduced_embed_path} is empty. Deleting directory.\n"
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
    if bool(embeds_dict.get("x")) and bool(embeds_dict.get("timestamp")):
        if not len(embeds_dict["x"]) == len(embeds_dict["timestamp"]):
            logger.warning(
                "\nThe lengths of timestamps and embeddings do not match. "
                "This could be the result of processing in multiple steps. "
                "It could also be caused if you are generating embeddings "
                "from annotations and the filenames in the csv file do not "
                "match the names of the audio files."
                "The safest way to avoid this, is by rerunning the dimensionality "
                "reduced embeddings. To do this delete the dim_reduced_embeddings folder.\n"
            )
    return embeds_dict


class EmbedAndLabelLoader:
    """
    Load and cache labels, dimensionally reduced embeddings and split data
    used for the embedding plots.
    """

    def __init__(self, dim_reduction_model, dashboard=False, **kwargs):
        """
        Initialize the embeddings and labels loader.

        Parameters
        ----------
        dim_reduction_model : str
            name of the dimensionality reduction model
        dashboard : bool
            whether the loader is used by the dashboard
        **kwargs
            additional keyword arguments (e.g., overwrite flag)
        """
        self.labels = dict()
        self.embeds = dict()
        self.split_data = dict()
        self.bool_noise = dict()
        self.dashboard = dashboard
        self.dim_reduction_model = dim_reduction_model
        self.kwargs = kwargs

    def get_data(self, model_name, label_by, remove_noise=False, **kwargs):
        """
        Load or return cached labels and embeddings for a model.

        Parameters
        ----------
        model_name : str
            name of the model
        label_by : str
            key of the metadata labels dict
        remove_noise : bool
            whether to filter out unannotated embeddings
        **kwargs
            additional keyword arguments passed to the path helpers

        Returns
        -------
        tuple of (dict, dict, dict)
            labels, embeddings, and data split by label
        """
        if not model_name in self.labels.keys():

            if not kwargs.get('widget_idx') is None and 'overwrite' in self.kwargs:
                self.kwargs['overwrite'] = False
            tup = get_labels_for_plot(model_name, **self.kwargs)
            self.labels[model_name], self.bool_noise[model_name] = tup

            dim_reduced_embed_path = le.get_dim_reduc_path_func(
                model_name,
                dim_reduction_model=self.dim_reduction_model,
                **kwargs,
            )

            self.embeds[model_name] = collect_dim_reduced_embeds(
                model_name,
                dim_reduced_embed_path,
                self.dim_reduction_model,
                **kwargs,
            )

        if remove_noise:
            return_embeds, return_labels = self.remove_noise_indices(
                model_name
            )
        else:
            return_labels = self.labels[model_name]
            return_embeds = self.embeds[model_name]
            return_embeds["index"] = np.arange(len(return_embeds["x"]))
            if len(return_embeds["metadata"]["audio_files"]) < len(
                return_embeds["x"]
            ):
                audiofilenames = []
                [
                    audiofilenames.extend([f] * nr)
                    for f, nr in zip(
                        return_embeds["metadata"]["audio_files"],
                        return_embeds["metadata"]["nr_embeds_per_file"],
                    )
                ]
                return_embeds["metadata"]["audio_files"] = audiofilenames

        if label_by in return_labels:
            return_splits = data_split_by_labels(
                return_embeds, return_labels[label_by], **kwargs
            )
        else:
            return [], [], {}
        return (
            # return_labels[label_by],
            return_labels,
            return_embeds,
            return_splits,
        )

    def remove_noise_indices(self, model_name):
        """
        Return labels and embeddings with unannotated points removed.

        Parameters
        ----------
        model_name : str
            name of the model

        Returns
        -------
        tuple of (dict, dict)
            filtered embeddings and filtered labels
        """
        return_labels, return_embeds = dict(), dict()
        bool_noise = self.bool_noise[model_name]

        for key, values in self.labels[model_name].items():
            if "noise" in key:
                return_labels[key] = values
            else:
                return_labels[key] = np.array(values, dtype=object)[
                    ~bool_noise
                ]

        for key, value in self.embeds[model_name].items():
            if not key == "metadata":
                return_embeds[key] = np.array(value)[~bool_noise]
            else:
                return_embeds["metadata"] = dict()
                for meta_key, meta_value in value.items():
                    if not isinstance(meta_value, list):
                        return_embeds["metadata"][meta_key] = meta_value
                    else:
                        if meta_key == "audio_files":
                            return_embeds["metadata"][meta_key] = np.array(
                                meta_value
                            )[~bool_noise]
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
        key of metadata_labels dict, by default "audio_file_name"
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
    labels, embeds, split_data = loader.get_data(
        model_name, label_by, **kwargs
    )

    fig, axes, return_axes = init_embed_figure(fig, axes, **kwargs)

    if len(labels[label_by]) == 0 and len(embeds) == 0:
        return fig

    if label_by == "audio_file_name":
        new_labels = [Path(l).stem + Path(l).suffix for l in labels]
        new_split_data = dict()
        for label in split_data.keys():
            new_label = Path(label).stem + Path(label).suffix
            new_split_data[new_label] = split_data[label]
        split_data = new_split_data

    c_label_dict = {
        lab: i for i, lab in enumerate(np.unique(labels[label_by]))
    }

    if return_axes:
        points = plot_embedding_points(
            axes, embeds, split_data, labels[label_by], c_label_dict, **kwargs
        )
        return axes, c_label_dict, points
    elif dashboard:
        return plot_embeddings_px(
            embeds, labels, label_by=label_by, **kwargs
        )
    else:
        set_colorbar_or_legend(
            fig, axes, points, c_label_dict, label_by=label_by, **kwargs
        )

        axes.set_title(f"{dim_reduction_model.upper()} embeddings")
        fig.savefig(paths.plot_path.joinpath("embeddings.png"), dpi=300)
        plt.close(fig)


def init_embed_figure(fig, axes, bool_3d=False, widget_idx=None, **kwargs):
    """
    Initialize a matplotlib figure and axes for embedding plots.

    Parameters
    ----------
    fig : plt.figure object or False
        existing figure handle
    axes : plt.axes object or False
        existing axes handle
    bool_3d : bool
        whether to create a 3D projection axes
    widget_idx : int or None
        figure number used for the dashboard widget
    **kwargs
        additional keyword arguments (unused)

    Returns
    -------
    tuple of (plt.figure, plt.axes, bool)
        figure handle, axes handle, and whether existing handles were used
    """
    if not fig:
        if bool_3d:
            fig, axes = plt.subplots(
                subplot_kw={"projection": "3d"}, figsize=(12, 8)
            )
        else:
            try:
                plt.close(widget_idx)
            except:
                pass
            fig = plt.figure(num=widget_idx, figsize=(12, 8), dpi=400)
            axes = fig.subplots()
        return_axes = False
    else:
        return_axes = True
    axes.set_xticks([])
    axes.set_yticks([])
    return fig, axes, return_axes


def get_boolean_array_for_annotated_embeddings(
    df_ground_truth, model_name, 
    ground_truth_files=None, gt_file=None,
    overwrite=False, **kwargs
):
    """
    Compute a boolean mask identifying embeddings that are annotated.

    Parameters
    ----------
    df_ground_truth : pandas.DataFrame
        ground truth annotations dataframe
    model_name : str
        name of the model
    ground_truth_files : list or None
        list of ground truth csv files for the model
    gt_file : pathlib.Path or None
        selected ground truth file
    overwrite : bool
        whether to force regeneration of the metadata labels
    **kwargs
        additional keyword arguments passed to metadata_labels

    Returns
    -------
    np.ndarray
        boolean array that is True for unannotated (noise) embeddings
    """
        
    df_metadata_labels = le.metadata_labels(
        model=model_name, overwrite=overwrite,
        return_type='dataframe', **kwargs
        )
    if kwargs.get('only_embed_annotations'):
        return np.array([False] * len(df_ground_truth))
    
    if not gt_file is None and not ground_truth_files is None:
        if (
            kwargs.get("label_column", settings.label_column) in str(gt_file) 
            or len(ground_truth_files) == 1
            ): 
            logger.info(
                f"Using {gt_file} to calculate the boolean "
                "mask to enable filtering of only annotated "
                "embeddings."
            )
        else:
            logger.info(
                "\nThere are multiple ground truth files: "
                "Since the label does not fit to the label "
                "label_column in settings.yaml It is unclear "
                "which one bacpipe should use "
                "to filter. Therefore the first file "
                f"{str(ground_truth_files[0])=} is selected. \n"
            )
        
    if max(df_ground_truth.simultaneous_labels) > 1:
        logger.warning(
            "\nYou have passed a multi-label ground truth array. "
            "However for visualization only one label will be displayed.\n"
        )
    df_metadata_labels['audiofilename'] = df_metadata_labels['audio_file_name']
    
    df_ground_truth = df_ground_truth[df_ground_truth.simultaneous_labels > 0]
    
    df_metadata_labels['start'] = [np.round(v, 4) for v in df_metadata_labels['start']]
    df_ground_truth['start'] = [np.round(v, 4) for v in df_ground_truth['start']]
    
    # Create multi-indexes for exact row matching
    meta_idx = pd.MultiIndex.from_frame(df_metadata_labels[['audiofilename', 'start']])
    gt_idx = pd.MultiIndex.from_frame(df_ground_truth[['audiofilename', 'start']])

    # Get your exact boolean mask
    is_in_ground_truth = meta_idx.isin(gt_idx)
    is_noise = ~is_in_ground_truth
    return is_noise


def get_single_label_gt_labels(df_ground_truth, bool_noise):
    """
    Reduce multi-label ground truth to a single label per segment.

    Parameters
    ----------
    df_ground_truth : pandas.DataFrame
        ground truth annotations dataframe
    bool_noise : np.ndarray
        boolean array that is True for unannotated (noise) embeddings

    Returns
    -------
    np.ndarray
        single label per embedding segment
    """
    if 'species_richness' in df_ground_truth.columns:
        df_ground_truth.rename(columns={'species_richness': 'simultaneous_labels'}, inplace=True)
        
    
    non_species_labels = [
        "start",
        "end",
        "audiofilename",
        "simultaneous_labels",
    ]
    
    # ensure that all segments where no annotations were associated with timestamps
    # are dropped
    df_ground_truth = df_ground_truth[df_ground_truth.simultaneous_labels > 0]
    
    # now filter the dataframe to only contain the species columns
    gt_without_metadata = df_ground_truth.drop(columns=non_species_labels)
    
    single_label = np.array(['noise'] * len(bool_noise), dtype='U50')
    
    # Now we need to ensure the length matches the other labels:
    assert (
        len(single_label[~bool_noise]) == len(gt_without_metadata.idxmax(axis=1).values)
        ), (
        "The lengths of the boolean array and the ground_truth arrays don't match. "
        "Try rerunning the script with overwrite=True."
        )
    
    # extract the species of the maximum index. now this is a bit
    # of a random selection becaues annotations are binary values
    # and so if there are 5 species present they are all equally
    # likely to be selected here
    single_label[~bool_noise] = gt_without_metadata.idxmax(axis=1).values
    return single_label

def get_labels_for_plot(model_name=None, overwrite=False, **kwargs):
    """
    Build the label dict and noise mask used for embedding plots.

    Parameters
    ----------
    model_name : str or None
        name of the model
    overwrite : bool
        whether to force regeneration of the metadata labels
    **kwargs
        additional keyword arguments passed to the label helpers

    Returns
    -------
    tuple of (dict, np.ndarray)
        labels by label key and the noise boolean mask
    """
    labels = dict()
    labels = le._get_metadata_labels(model_name, overwrite=overwrite, return_type='dict', **kwargs)

    paths = le.get_paths(model_name)
    ground_truth_files = list(
        paths.labels_path.glob("ground_truth*csv")
    )
    if len(ground_truth_files) > 0:        
        for gt_file in ground_truth_files:
            try:
                ground_truth_df = le.get_ground_truth(
                    model_name, file_path=gt_file, return_type="dataframe"
                )
                
                bool_noise = get_boolean_array_for_annotated_embeddings(
                    ground_truth_df, model_name,
                    gt_file=gt_file, ground_truth_files=ground_truth_files, 
                    **kwargs
                )
                label = gt_file.stem.replace("ground_truth_", "")
                
                labels[label] = get_single_label_gt_labels(
                    ground_truth_df, bool_noise
                    )
            except Exception as e:
                logger.warning(
                    "\nBuilding of ground truth labels for plots failed "
                    f"due to {str(e)}. Continuing without ground truth labels. \n"
                )
                bool_noise = np.array([False] * len(list(labels.values())[0]))
                

        
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
                    if len(values) == len(bool_noise):
                        labels[name] = values
                    elif 'no_noise' in name:
                        if len(values) == len(np.where(~bool_noise)[0]):
                            labels[name] = values
                    else:
                        logger.warning(
                            f"\nThe clustering {name} does not match the length "
                            "of generated embeddings and can therefore not be "
                            "correctly displayed.\n"
                        )
                        labels[name] = np.array(
                            ["noise"] * len(bool_noise), dtype=object
                        )
    return labels, bool_noise


def set_colorbar_or_legend(
    fig, axes, points, c_label_dict, label_by, **kwargs
):
    """
    Add a colorbar or a legend to the embedding plot depending on label count.

    Parameters
    ----------
    fig : plt.figure object
        figure handle
    axes : plt.axes object
        axes handle
    points : list
        plt point objects for the legend or colorbar
    c_label_dict : dict
        mapping of label name to label index
    label_by : str
        key of the label dict used for coloring
    **kwargs
        additional keyword arguments passed to set_legend

    Returns
    -------
    tuple of (plt.figure, plt.axes)
        updated figure and axes handles
    """
    if len(c_label_dict.keys()) > kwargs.get(
        "max_nr_categories", settings.max_nr_categories
    ):
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
        fig, axes = set_legend(
            hands, labs, fig, axes, label_by=label_by, **kwargs
        )
    return fig, axes


def plot_embedding_points(
    axes,
    embeds,
    split_data,
    labels,
    c_label_dict,
    remove_noise=False,
    **kwargs,
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
    if len(c_label_dict.keys()) > kwargs.get(
        "max_nr_categories", settings.max_nr_categories
    ):
        import matplotlib.cm as cm

        cmap = cm.viridis  # or 'plasma', 'inferno', 'magma', etc.
        # if remove_noise:
        #     bool_labels = np.array(labels) != "noise"
        #     labels = np.array(labels)[bool_labels]
        # else:
        #     bool_labels = [True] * len(labels)

        num_labels = np.array([c_label_dict[lab] for lab in labels])
        if not len(labels) == len(embeds["x"]):
            raise AssertionError(
                f"\nThe number of labels is {len(labels)} whereas the number of "
                f"embedding points is {len(embeds['x'])}. This mismatch could "
                "be the result of an incomplete run and bacpipe is using "
                "the dim_reduced_embeddings corresponding to that. Check if in your results folder "
                "there are not multiple dim_reduced_embeddings, and if so, delete the incomplete one.\n"
            )
        if len(np.array(embeds["x"]).shape) > 1:
            embeds["x"] = (np.array(embeds["x"])[:, 0],)
            embeds["y"] = (np.array(embeds["y"])[:, 0],)
        points = axes.scatter(
            # np.array(embeds["x"])[bool_labels],
            # np.array(embeds["y"])[bool_labels],
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
    handles,
    labels,
    fig,
    axes,
    bool_plot_centroids=False,
    dashboard=False,
    label_by=None,
    **kwargs,
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
    label_by : str, optional
        key of the label dict used for coloring. Audio file names are long, so
        their legend is forced to a single column to avoid taking up too much
        horizontal space.

    Returns
    -------
    plt.fig object
        figure handle
    plt.axes object
        axes handle
    """

    # Calculate number of columns dynamically based on the number of labels
    num_labels = len(labels)  # Number of labels in the legend
    ncol = min(
        num_labels, 5
    )  # Use 6 columns or fewer if there are fewer labels

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
        # Compute the column count so the legend
        # stays inside the figure boundaries even when there are many labels (e.g. many species). 
        num_labels = len(new_labels)
        fig_w, fig_h = fig.get_size_inches()
        max_rows = max(1, int(fig_h / 0.28))
        ncol = max(1, int(np.ceil(num_labels / max_rows)))
        max_cols = max(1, 2)#int((0.45 * fig_w) / 0.7))
        ncol = min(ncol, max_cols)
        if label_by == "audio_file_name":
            ncol = 1

        fontsize = 6 if num_labels > 40 else 7
        markerscale = 3 if num_labels > 40 else 4

        # Reserve only as much horizontal space as the legend actually needs
        right = 0.8 - min(0.5, (ncol * 0.7) / fig_w)

        fig.subplots_adjust(right=right)
        fig.tight_layout(
            rect=(0.0, fig.subplotpars.bottom, right, fig.subplotpars.top)
        )

        fig.legend(
            new_handles,
            new_labels,
            loc="outside right",
            ncol=ncol,
            markerscale=markerscale,
            fontsize=fontsize,
            frameon=False,
        )
    else:

        if label_by == "audio_file_name":
            ncol = 1
        fig.subplots_adjust(bottom=0.2)
        fig.legend(
            new_handles,
            new_labels,  # Use the handles and labels from the plot
            loc="outside lower center",  # Center the legend
            ncol=ncol,  # Number of columns
            markerscale=6,
        )
    return fig, axes


def data_split_by_labels(embeds_dict, labels, **kwargs):
    """
    Split data by labels for scatterplots.

    Parameters
    ----------
    embeds_dict : dict
        embeddings by model
    labels : list
        list of labels
    **kwargs : dict
        Explicitly passed kwargs override the defaults from
        ``bacpipe/settings.yaml``, e.g. ``max_nr_categories``.

    Returns
    -------
    dict
        x and y data corresponding to labels
    """
    split_data = {}
    uni_labels = np.unique(labels)
    if len(uni_labels) > kwargs.get(
        "max_nr_categories", settings.max_nr_categories
    ):
        split_data["all"] = np.array(
            [
                np.array(embeds_dict["x"]),
                np.array(embeds_dict["y"]),
            ]
        )
    else:
        for label in uni_labels:
            split_data[str(label)] = np.array(
                [
                    np.array(embeds_dict["x"])[np.array(labels) == label],
                    np.array(embeds_dict["y"])[np.array(labels) == label],
                ]
            )

    return split_data


def return_rows_cols(num):
    """
    Determine the grid dimensions for a comparison plot.

    The grid is chosen so that no subplot slots are left empty: for small
    model counts the grid matches the model count exactly. This keeps the
    individual plots as large as possible and avoids a dead band in the
    comparison figure (which previously always used a 1x3 grid for up to
    three models, leaving a third of the width empty when only two models
    were compared).

    Parameters
    ----------
    num : int
        number of subplots to lay out

    Returns
    -------
    tuple of (int, int)
        number of rows and columns for the grid
    """
    if num <= 3:
        return 1, max(2, num)
    elif num == 4:
        return 2, 2
    elif num <= 6:
        return 2, 3
    elif num <= 9:
        return 3, 3
    elif num <= 12:
        return 3, 4
    elif num <= 16:
        return 4, 4
    elif num <= 20:
        return 4, 5
    else:
        return 5, int(np.ceil(num / 5))


def set_figsize_for_comparison(rows, cols):
    """
    Choose a figure size based on the comparison grid dimensions.

    Parameters
    ----------
    rows : int
        number of grid rows
    cols : int
        number of grid columns (unused in the size selection)

    Returns
    -------
    tuple of (float, float)
        figure width and height in inches
    """
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

    fig.suptitle(
        f"Comparison of {dim_reduction_model} embeddings", fontweight="bold"
    )
    if not dashboard:
        fig.savefig(plot_path.joinpath("comp_fig.png"), dpi=300)
        plt.close(fig)
    else:
        return fig
    
def get_arrays_for_spectrogram_text(labels, label_by, data_dict, embeds, **kwargs):
    """
    Build extra label arrays shown in the spectrogram hover text.

    Parameters
    ----------
    labels : dict
        labels by label key
    label_by : str
        key of the label dict currently used for coloring
    data_dict : dict
        data arrays already included in the figure
    embeds : dict
        embeddings dict with metadata
    **kwargs : dict
        Explicitly passed kwargs override the defaults from
        ``bacpipe/settings.yaml``, e.g. ``metadata_label_keys``,
        ``evaluations_dir`` and ``nr_predictions_to_display``.

    Returns
    -------
    dict
        additional label arrays for the hover text
    """
    dlk = kwargs.get("metadata_label_keys", settings.metadata_label_keys)
    label_copy = labels.copy()
    # remove clustering labels from dict
    
    for label_key in labels.keys():
        if 'no_noise' in label_key:
            label_copy.pop(label_key)
            
        
    df_lab = {}
    for k, v in label_copy.items():
        if (
            not k in dlk 
            and not k == label_by
            ):
            df_lab[k] = list(v)
    [df_lab.pop(k) for k in data_dict.keys() if k in df_lab.keys()]
    
    if 'default_classifier' in label_copy:
        file_paths = list((
            Path(embeds['metadata']['embed_dir'])
            .parent
            .parent
            / kwargs.get("evaluations_dir", settings.evaluations_dir)
            / embeds['metadata']['model_name']
            / 'predictions'
            ).glob('*all_predictions*'))
        if len(file_paths) > 0:
            if 'csv' in str(file_paths[0]):
                try:
                    all_preds = pd.read_csv(file_paths[0], index_col=False)
                except:
                    all_preds = None
            elif 'parquet' in str(file_paths[0]):
                try:
                    all_preds = pd.read_parquet(file_paths[0])
                except:
                    all_preds = None
        try:
            ## now filter the df so we only have the top 5 preds
            just_labels = all_preds.drop(columns=['audiofilename', 'start', 'end', 'simultaneous_labels'])
            if 'Unnamed: 0' in just_labels.columns:
                just_labels = just_labels.drop(columns=['Unnamed: 0'])
            np_labels = just_labels.values.T
            
            k=kwargs.get(
                "nr_predictions_to_display", settings.nr_predictions_to_display
            )
            top_k_indices = np.argsort(np.array(np_labels), axis=0)[-k:][::-1]
            top_k_probs = np.sort(np_labels, axis=0)[-k:][::-1]
            top_k_species = just_labels.columns.values[top_k_indices].T
            
            top_k_probs = top_k_probs.T
            top_k_species[top_k_probs == 0] = ''
            
            i = 0
            species, probs = [], []
            for idx, label in enumerate(label_copy['default_classifier']):
                if label == 'below_thresh':
                    species.append([])
                    probs.append([])
                else:
                    species.append(top_k_species[i].tolist())
                    probs.append(top_k_probs[i].tolist())
                    i += 1
            df_lab[f'top_{k}_species'] = species
            df_lab[f'top_{k}_confidence'] = probs
        except Exception as e:
            logger.info(
                f"\nTop {k} predictions for display could not be loaded. "
                "The reason could be that a previous run failed and not all "
                "predictions were saved. Regenerating the embeddings is the "
                f"best chance of getting this to work. \n{str(e)}\n"
            )
    return df_lab


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
    clust_dict = json.load(
        open(plot_path.joinpath("clustering_results.json"), "r")
    )
    new_order = dict(
        sorted(
            clust_dict.items(),
            key=lambda kv: kv[1]["ARI"][order_metric],
            reverse=True,
        )
    )
    positions = {
        mod: ax.get_position() for mod, ax in zip(new_order, axes.flatten())
    }
    for model, ax in zip(models, axes.flatten()):
        if not model in positions.keys():
            continue
        ax.set_position(positions[model])


def plot_embeddings_px(
    embeds, labels, label_by="label", **kwargs
):
    """
    Create a plotly embedding scatter plot.

    Parameters
    ----------
    embeds : dict
        embeddings dict with x, y (and optional z) arrays and metadata
    labels : dict
        labels by label key
    label_by : str
        key of the label dict used for coloring
    **kwargs
        additional keyword arguments (e.g., color_continuous)

    Returns
    -------
    plotly.graph_objects.Figure
        embedding scatter plot figure
    """
    # 1. Prepare Data
    if len(np.array(embeds["x"]).shape) > 1:
        embeds["x"] = np.array(embeds["x"]).squeeze()
        embeds["y"] = np.array(embeds["y"]).squeeze()
    x_data = embeds["x"]
    y_data = embeds["y"]
    if not embeds.get('z') is None:
        z_data = embeds.get('z')

    audiofilenames = embeds["metadata"]["audio_files"]

    starts = embeds["timestamp"]

    if "durations" in embeds.keys() and len(embeds.get("durations")) > 0:
        ends = np.array(embeds.get("durations")) + np.array(starts)
        ends = ends.tolist()
    else:
        ends = np.array(starts) + (
            embeds["metadata"]["segment_length (samples)"]
            / embeds["metadata"]["sample_rate (Hz)"]
        )
        ends = ends.tolist()

    starts, ends = np.round(starts, 4), np.round(ends, 4)

    # Calculate unique labels to decide on Legend vs Colorbar
    unique_labels = np.unique(labels[label_by])
    n_labels = len(unique_labels)

    # When the number of categories is below the settings threshold we want
    # to show a discrete legend. Cluster labels (e.g. kmeans) are typically
    # integers, but plotly express treats numeric "color" columns as a
    # continuous variable and would draw a colorbar instead of a legend.
    # Casting the label values to strings keeps them categorical.
    if n_labels <= kwargs.get(
        "max_nr_categories", settings.max_nr_categories
    ):
        labels_for_plot = [str(lbl) for lbl in labels[label_by]]
        unique_labels = np.unique(labels_for_plot)
        n_labels = len(unique_labels)
    else:
        labels_for_plot = labels[label_by]

    # Create an integer mapping for high-cardinality plotting
    # (Plotly needs numbers to generate a gradient colorbar)
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    label_ids = [label_to_id[l] for l in labels_for_plot]

    data_dict = {
        "x": x_data,
        "y": y_data,
        "label": labels_for_plot,  # The actual string (for hover/legend)
        "label_id": label_ids,  # The integer (for colorbar)
        "audiofilename": audiofilenames,
        "start": starts,
        "end": ends,
        "idx": embeds["index"],
        "model": [embeds['metadata']['model_name']]*len(x_data),
    }
    
    if not embeds.get('z') is None:
        data_dict['z'] = z_data
        
    df_lab = get_arrays_for_spectrogram_text(
        labels, label_by, data_dict, embeds, **kwargs
        )
    from bacpipe.embedding_evaluation.clustering.cluster import convert_numpy_types
    # Pack variable labels as JSON string to preserve order and labels
    data_dict["variable_labels_json"] = (
        [
            json.dumps({k: convert_numpy_types(v) for k, v in zip(df_lab.keys(), row)})
            for row in zip(*df_lab.values())
        ]
        if df_lab
        else [json.dumps({})] * len(labels[label_by])
    )

    data_dict = {**data_dict}

    df = pd.DataFrame(data_dict)
    df = df.sort_values("label")

    hover_data = {k: False for k in data_dict}
    for k in hover_data.keys():
        if k in ["label", "audiofilename", "start", "end"]:
            hover_data[k] = True

    custom_data = [
        "audiofilename",
        "start",
        "end",
        "idx",
        "label",
        "variable_labels_json",
        "label_id",
        "model",
    ]

    # 2. Setup Figure based on Label Count
    if n_labels > kwargs.get(
        "max_nr_categories", settings.max_nr_categories
    ):
        if not embeds.get('z') is None:
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                size_max=1,
                color="label_id",
                hover_data=hover_data,
                custom_data=custom_data,
                title=f"Embedding Plot - {embeds['metadata']['model_name']} - {label_by}",
                color_continuous_scale=kwargs.get("color_continuous"),
            )
        else:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="label_id",
                hover_data=hover_data,
                custom_data=custom_data,
                title=f"Embedding Plot - {embeds['metadata']['model_name']} - {label_by}",
                render_mode="webgl",
                color_continuous_scale=kwargs.get("color_continuous"),
            )

        tick_vals = (
            np.linspace(0, n_labels - int(n_labels // 100 + 1), 6)
            .astype(int)
            .tolist()
        )
        tick_text = [str(unique_labels[i]) for i in tick_vals]

        fig.update_coloraxes(
            colorbar_title=label_by,
            colorbar_tickmode="array",
            colorbar_tickvals=tick_vals,
            colorbar_ticktext=tick_text,
        )

    else:
        if not embeds.get('z') is None:
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                size_max=1,
                color="label",
                hover_data=hover_data,
                custom_data=custom_data,
                title=f"Embedding Plot - {embeds['metadata']['model_name']} - {label_by}",
                color_discrete_sequence=COLOR_DISCRETE,
            )
        else:
            # force a discrete legend
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="label",
                hover_data=hover_data,
                custom_data=custom_data,
                title=f"Embedding Plot - {embeds['metadata']['model_name']} - {label_by}",
                render_mode="webgl",
                color_discrete_sequence=COLOR_DISCRETE,
            )

        # Configure the Discrete Legend
        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0,
                xanchor="left",
                x=1.02,
                title_text=label_by,
            )
        )

    fig.update_layout(
        # autosize=True,
        uirevision=True,
        scene=dict(uirevision=True),
        template="plotly_white",
        height=kwargs.get("embed_fig_height", settings.embed_fig_height),
        clickmode="event",
        hovermode="closest",
        # margin=dict(l=20, r=20, t=40, b=20),
        margin=dict(l=0, r=80, t=40, b=0),
        # Ensure selection tools are available
        modebar=dict(add=["lasso2d", "select2d"], remove=["autoScale2d"]),
    )
    marker_sz = 3 if embeds.get("z") is not None else 8
    fig.update_traces(marker_size=marker_sz, marker_opacity=0.6)
    return fig
