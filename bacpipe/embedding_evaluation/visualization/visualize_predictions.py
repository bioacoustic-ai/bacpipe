import json

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import plotly.express as px
import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from bacpipe.embedding_evaluation.probing.inference_probe import (
    prepare_probe_inference,
    run_probe_inference,
)


def plot_classification_results(
    task_name,
    paths=None,
    results=None,
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
    results : dict
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
    if not results:
        probe_path = paths.probe_path / f"probe_results_{task_name}.json"
        if not probe_path.exists():
            error = (
                f"\nThe probing file {probe_path} does not exist. Perhaps it was not "
                "created yet. To avoid getting this error, make sure you have not "
                " included 'probing' in the 'evaluation_tasks'. If you want to compute "
                "probing, make sure to set `overwrite=True`."
            )
            logger.exception(error)
            raise AssertionError(error)

        with open(
            paths.probe_path / f"probe_results_{task_name}.json", "r"
        ) as f:
            results = json.load(f)

    # Filter overall results if needed
    results["overall"] = {
        k: v for k, v in results["overall"].items() if not "micro" in k
    }

    # Sort classes by accuracy for better visualization
    class_items = sorted(
        results["per_class_accuracy"].items(), key=lambda x: x[1], reverse=True
    )
    class_names = [item[0] for item in class_items]
    class_values = [item[1] for item in class_items]

    # Set figure size based on number of classes and return_fig
    if return_fig:
        # For dashboard, make height adapt to number of classes
        height = max(4, len(class_names) * 0.22)
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

    # Create results string
    results_string = "".join(
        [f"{k}: {v:.3f} | " for k, v in results["overall"].items()]
    )

    fig.suptitle(
        f"Classwise accuracy for {task_name} "
        f"probe with {model_name.upper()}\n"
        f"{results_string}",
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
        path.joinpath(f"probe_results_{task_name}_{model_name}.png"),
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
    results = {}
    for model_name in model_list:
        paths = path_func(model_name)
        if task == "clustering":
            key = "clust_path"
        elif task == "probing":
            key = "probe_path"
        for file in getattr(paths, key).rglob("*results*.json"):
            if task == "probing":
                subtask = file.stem.split("_")[-1]
                results[f"{model_name}({subtask})"] = json.load(
                    open(file, "r")
                )
            else:
                results[model_name] = json.load(open(file, "r"))
    return results


def plot_per_class_results(plot_path, task_name, model_list, results):
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
    results : dict
        performance dictionary
    """
    per_class_results = {
        m: v["per_class_accuracy"] for m, v in results.items()
    }
    overall_results = {m: v["overall"] for m, v in results.items()}
    num_classes = len(per_class_results[model_list[0]].keys())
    fig_width = max(12, num_classes * 0.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

    cmap = plt.cm.tab10
    model_colors = cmap(np.arange(len(model_list)) % cmap.N)

    d = {m: v["macro_accuracy"] for m, v in overall_results.items()}
    model_list = sorted(d, key=d.get, reverse=True)
    all_classes = sorted(per_class_results[model_list[0]].keys())

    for i, model_name in enumerate(model_list):
        class_values = per_class_results[model_name].values()

        ax.scatter(
            np.arange(len(class_values)),
            class_values,
            color=model_colors[i],
            label=f"{model_name.upper()} "
            + f"(accuracy: {overall_results[model_name]['macro_accuracy']:.3f})",
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
        f"Per class results for {task_name} across models",
        fontsize=14,
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Classes")
    ax.set_xticks(np.arange(len(all_classes)))
    ax.set_xticklabels(all_classes, rotation=90)

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.05, 1), title="Models", fontsize=10
    )

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
from bacpipe.embedding_evaluation.visualization.visualize_spectrograms import (
    SpectrogramPlot,
)


def plot_classification_heatmap(
    event,
    predictions_loader,
    model,
    accumulate_by,
    threshold,
    species=None,
    **kwargs,
):
    """
    Generate a presence heatmap for a species across time and hours.

    Parameters
    ----------
    event : object or None
        widget event triggering the update
    predictions_loader : PredictionsLoader object
        loader providing the binary presence data
    model : str
        name of the model
    accumulate_by : str
        time unit used to aggregate ("day", "week", or "month")
    threshold : float or str
        detection threshold
    species : str or None
        species to plot, by default None
    **kwargs
        additional keyword arguments (e.g., heatmap_fig_height)

    Returns
    -------
    plotly.graph_objects.Figure
        presence heatmap figure
    """
    if event is None and species is None:
        return SpectrogramPlot.dummy_image(
            title="Click the button to generate a prediction heatmap."
        )
    try:
        predictions_loader.get_data(model, threshold, **kwargs)
        accumulated_presence = predictions_loader.accumulate_data(
            species, accumulate_by
        )
        timestamps = predictions_loader.timestamps
    except Exception as e:
        logger.exception(e)
        return SpectrogramPlot.dummy_image(title=str(e))
    logger.info("Redrawing heatmap plot")

    # Prepare data - mask values below 0
    plot_data = accumulated_presence.T.copy()
    plot_data = np.where(plot_data < 0, np.nan, plot_data)

    # Get time labels based on accumulation type
    if accumulate_by == "day":
        y_labels = [str(ts.date()) for ts in timestamps]
        y_axis_label = "Dates"
    elif accumulate_by == "month":
        y_labels = [f"{date.year}-{date.month}" for date in timestamps]
        y_axis_label = "Months"
    elif accumulate_by == "week":
        y_labels = [
            f"{date.year}-W{date.isocalendar().week}" for date in timestamps
        ]
        y_axis_label = "Weeks"

    # Hour labels (0-23)
    x_labels = list(range(24))

    # Get unique labels (preserving order)
    unique_labels = []
    seen = set()
    for label in y_labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)
    y_indices = list(range(len(unique_labels)))
    nr_y_ticks = max(
        1, int(kwargs.get("heatmap_fig_height", 600) / 100)
    )

    clfier_type = (
        f"{predictions_loader.current_clfier_type} probing"
        if not predictions_loader.current_clfier_type == "Integrated"
        else "integrated classifier"
    )
    # Create heatmap
    fig = px.imshow(
        plot_data,
        labels=dict(
            x="Hours", y=y_axis_label, color="Binary presence per hour"
        ),
        x=x_labels,
        # y=np.unique(y_labels),
        y=y_indices,  # ✅ Use integer indices instead of labels
        color_continuous_scale="Viridis",
        zmin=0,  # Values below this will be white (nan handling)
        aspect="auto",
        title=(
            f"Presence heatmap using {model.upper()} with "
            f"{clfier_type} <br>"
            f"for {species} "
            f"with threshold of {PredictionsLoader.verify_threshold(threshold)}."
        ),
    )

    # Customize layout
    fig.update_layout(
        # autosize=True,
        # width=600,
        height=kwargs.get("heatmap_fig_height"),
        template="plotly_white",
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 6, 12, 18, 23],
            ticktext=["0", "6", "12", "18", "23"],
        ),
        # yaxis=dict(
        #     autorange='reversed'  # Optional: match seaborn orientation
        # ),
        yaxis=dict(
            autorange="reversed",  # Match seaborn orientation
            tickmode="array",
            tickvals=y_indices[::nr_y_ticks],
            ticktext=unique_labels[::nr_y_ticks],
        ),
        coloraxis_colorbar=dict(
            title="",
        ),
        annotations=[
            dict(
                text="Binary presence per hour",
                textangle=-90,  # This is the magic 90-degree rotation
                xref="paper",  # Position relative to the whole figure
                yref="paper",
                x=1.2,  # Adjust this to move it left/right of the colorbar
                y=0.5,  # Center it vertically
                showarrow=False,
                font=dict(size=14),
            )
        ],
        margin=dict(r=100),
    )

    # Make NaN values appear white
    fig.update_traces(
        hovertemplate="Hour: %{x}<br>"
        + y_axis_label
        + ": %{y}<br>Presence: %{z}<extra></extra>",
        customdata=np.array(unique_labels)[
            :, None
        ],  # Add date labels to hover
    )

    return fig


class PredictionsLoader:
    """
    Load and cache binary presence data, ground truth and classifier
    predictions for the predictions pages.
    """

    def __init__(
        self,
        vis_loader,
        path_func,
        models,
        panel_selection,
        progress_bar,
        loading_pane,
        thresh=0.5,
    ):
        """
        Initialize the predictions loader.

        Parameters
        ----------
        vis_loader : EmbedAndLabelLoader object
            loader providing the embeddings
        path_func : function
            function returning the paths for a given model
        models : list
            list of models
        panel_selection : object
            dropdown widget for selecting the species
        progress_bar : object
            progress bar widget
        loading_pane : object
            pane widget for status messages
        thresh : float
            default detection threshold
        """
        self.vis_loader = vis_loader
        self.path_func = path_func
        self.models = models
        self.thresh = thresh
        self.panel_selection = panel_selection
        self.progress_bar = progress_bar
        self.loading_pane = loading_pane

    def get_data(
        self, model, threshold, clfier_type=None, probe_path="", **kwargs
    ):
        """
        Load or recompute the binary presence data for a model.

        Parameters
        ----------
        model : str
            name of the model
        threshold : float or str
            detection threshold
        clfier_type : str or None
            type of classifier ("Linear", "Integrated", or None)
        probe_path : str
            path to the probe used for the linear classifier
        **kwargs
            additional keyword arguments (unused)

        Returns
        -------
        None
            results are cached on the instance
        """
        threshold = self.verify_threshold(threshold)

        # If no trained probe exists, fall back to the integrated classifier.
        if clfier_type is None:
            clfier_type = getattr(self, "current_clfier_type", "Integrated")
        if not (
            self.path_func(self.models[0]).probe_path / "linear_probe.pt"
        ).exists():
            logger.warning(
                "\nNo Linear probe has been trained yet, therefore Linear is not "
                "an option. Enable probing to ensure a linear classifier is saved "
                "first. Then it can be used here.\n"
            )
            clfier_type = "Integrated"

        # Serve the cached result only if it matches the current request and
        # is fully consistent. A failed run must not leave a stale cache
        # behind, which is why the "overall" column is part of the check
        # (it is only added after a fully successful load).
        if hasattr(self, "binary_presence") and (
            self.current_model == model
            and self.current_threshold == threshold
            and self.current_clfier_type == clfier_type
            and self.class_dict is not None
            and "overall" in self.class_dict
        ):
            return

        # Only touch the cached state after everything succeeded so that a
        # failed run cannot leave stale/inconsistent data in the cache.
        try:
            if clfier_type == "Linear":
                self.loading_pane.value = "Loading embeddings for classification"
                if not probe_path:
                    probe_path = self.path_func(model).probe_path / "linear_probe.pt"
                linear_probe, class_dict = prepare_probe_inference(
                    model, probe_path, **kwargs
                )
                self.loading_pane.value = "Running linear probe"
                threshold = self.verify_threshold(threshold)

                binary_presence = run_probe_inference(
                    model,
                    linear_probe,
                    threshold,
                    return_binary_presence=True,
                    callbacks={"progress_bar": self.progress_bar},
                    audio_dir=self.path_func(model).audio_dir,
                    main_results_dir=self.path_func(model).main_results_dir,
                    **kwargs,
                )

            elif clfier_type == "Integrated":
                self.loading_pane.name = "Preparing heatmap"
                self.loading_pane.value = "Loading precomputed embeddings"
                binary_presence, class_dict = self.load_classification(
                    model, threshold
                )

            else:
                raise ValueError(
                    f"\nUnknown classifier type: {clfier_type!r}\n"
                )

            self.embed_dict = self.vis_loader.embeds[model]

            if binary_presence is None:
                warning_string = (
                    "\nIt seems like the classifier hasn't been run yet, or <br>"
                    f"that {model} doesn't have a pretrained classifier. <br>"
                    "If the model has a pretrained classifier, please rerun <br>"
                    "bacpipe with the setting `run default classifier` set to `True`.\n"
                )
                self.loading_pane.value = warning_string
                raise FileNotFoundError(warning_string)

            if not len(self.embed_dict["x"]) == len(binary_presence):
                logger.warning(
                    "\nThere is a mismatch between the number of embeddings "
                    "and the number of predictions. Going to zero pad the "
                    "rest, but this could misalign things. \n"
                )
                binary_presence = np.pad(
                    binary_presence,
                    (
                        (0, len(self.embed_dict["x"]) - len(binary_presence)),
                        (0, 0),
                    ),
                    "constant",
                )

            self.get_timestamps_per_embedding(model)

            class_dict["overall"] = len(class_dict)
            binary_presence = np.concatenate(
                [
                    binary_presence.T,
                    [np.sum(binary_presence, axis=1).astype(np.int8)],
                ]
            ).T

            class_dict = self.reorder_by_most_occurrance(
                binary_presence, class_dict
            )
        except Exception:
            self.binary_presence = None
            self.class_dict = None
            raise

        # Commit to the cache only on success.
        self.binary_presence = binary_presence
        self.class_dict = class_dict
        self.current_model = model
        self.current_threshold = threshold
        self.current_clfier_type = clfier_type
        self.panel_selection.options = list(self.class_dict.keys())

    @staticmethod
    def verify_threshold(threshold):
        """
        Normalize a threshold widget value to a float.

        Parameters
        ----------
        threshold : float or str
            raw threshold value, possibly an empty string

        Returns
        -------
        float
            validated threshold
        """
        if threshold == "":
            threshold = 0.5
        else:
            threshold = float(threshold)
        return threshold

    @staticmethod
    def reorder_by_most_occurrance(probs, label2index):
        """
        Reorder the class index mapping by total occurrence count.

        Parameters
        ----------
        probs : np.ndarray
            binary presence array of shape (n_embeddings, n_classes)
        label2index : dict
            mapping of class name to column index

        Returns
        -------
        dict
            class to index mapping sorted by decreasing occurrence
        """
        sums = [sum(probs[:, a]) for a in range(probs.shape[1])]

        sorted_l2i = dict(
            sorted(label2index.items(), key=lambda x: sums[x[1]], reverse=True)
        )
        return sorted_l2i

    def get_classes(self, path):
        """
        Load the class names from a probe's label2index file.

        Parameters
        ----------
        path : str or pathlib.Path
            path to the probe file

        Returns
        -------
        list
            class names, or an empty list if no probe exists
        """
        if path == "":
            path = (
                self.path_func(self.models[0]).probe_path / "linear_probe.pt"
            )
        if path.exists():
            with open(Path(path).parent / "label2index.json", "r") as f:
                classes = json.load(f)
            return list(classes.keys())
        else:
            return []

    def load_classification(self, model, threshold):
        """
        Load the integrated classifier outputs as binary presence data.

        Parameters
        ----------
        model : str
            name of the model
        threshold : float
            detection threshold applied to the class probabilities

        Returns
        -------
        tuple of (np.ndarray or None, dict or None)
            binary presence array and class to index mapping, or
            (None, None) if no classifier outputs exist
        """
        integrated_clfier_path = self.path_func(model).preds_path.joinpath(
            "original_classifier_outputs"
        )
        if not integrated_clfier_path.exists():
            return None, None
        else:
            files = list(integrated_clfier_path.rglob("*json"))

        if not (integrated_clfier_path / "as_dataframe.parquet").exists():
            cl_dict = {}
            total_length = 0
            keys2idx = {}
            for idx, file in enumerate(files):
                with open(file, "r") as f:
                    d = json.load(f)
                    current_time_bins = d["head"]["Time bins in this file"]
                    d.pop("head")

                    for k, v in d.items():
                        cl_dict[k] = np.zeros(
                            [total_length + current_time_bins]
                        )
                        if not keys2idx:
                            keys2idx[k] = 0
                        if not k in keys2idx:
                            keys2idx[k] = max(keys2idx.values()) + 1

                        cl_dict[k][
                            np.array(v["time_bins_exceeding_threshold"])
                            + total_length
                        ] = v["classifier_predictions"]
                        # file_specific_classification[v['time_bins_exceeding_threshold'], k2idx[k]] = v['classifier_predictions']
                    for species in [
                        k
                        for k, v in cl_dict.items()
                        if len(v) < total_length + current_time_bins
                    ]:
                        cl_dict[species] = np.hstack(
                            [cl_dict[species], np.zeros([current_time_bins])]
                        )

                    total_length += current_time_bins
                self.progress_bar.value = int((idx + 1) / len(files) * 100)
            if len(keys2idx) == 0:
                error_string = (
                    "\nNo predictions have been found in the provdided data "
                    "using this model. Please try again with a different "
                    "threshold or different model. \n"
                    "Simply changing the threshold will not change this, "
                    "given that there were no predictions with the minimum "
                    "threshold, the classifications need to be recomputed. "
                    "The easiest way to do this is to delete the generated "
                    "classifications which will force a recomputation.\n"
                )
                logger.exception(error_string)
                raise ValueError(
                    error_string
                )
            import pandas as pd

            probs_array = np.array(list(cl_dict.values())).T
            df = pd.DataFrame(probs_array)
            df.columns = keys2idx.keys()
            df.to_parquet(integrated_clfier_path / "as_dataframe.parquet")
        else:
            import pandas as pd

            df = pd.read_parquet(
                integrated_clfier_path / "as_dataframe.parquet"
            )
            keys = df.columns
            keys2idx = {k: i for i, k in enumerate(keys)}
        # binary_classification = probs_array[probs_array > thresh]
        binary_classification = np.zeros(df.shape, dtype=np.int8)
        binary_classification[df > threshold] = 1

        return binary_classification, keys2idx

    def accumulate_data(self, species, accumulate_by="day"):
        """
        Accumulate the binary presence of a species into a time heatmap.

        Parameters
        ----------
        species : str or None
            species to accumulate, or None for the overall presence
        accumulate_by : str
            time unit used to aggregate ("day", "week", or "month")

        Returns
        -------
        np.ndarray
            accumulated presence array of shape (24, n_time_bins)
        """
        if not species:
            species = "overall"
        # Fall back to the overall presence instead of crashing.
        if species not in self.class_dict:
            logger.warning(
                f"\nSpecies {species!r} not found in the current classifier "
                "outputs, falling back to 'overall'.\n"
            )
            species = "overall"
        self.panel_selection.value = species
        species_idx = self.class_dict[species]
        species_presence = self.binary_presence[:, species_idx]

        dates = np.array([ts.date() for ts in self.timestamps])
        hours = np.array([ts.hour for ts in self.timestamps])
        if accumulate_by == "day":
            date_tuple = [(d.year, d.month, d.day) for d in dates]
            accumulated = self.transform_presence_into_hour_heatmap(
                species_presence, hours, accumulator=date_tuple
            )
        elif accumulate_by == "week":
            week_tuple = [
                (date.year, date.isocalendar().week) for date in dates
            ]
            accumulated = self.transform_presence_into_hour_heatmap(
                species_presence, hours, accumulator=week_tuple
            )
        elif accumulate_by == "month":
            month_tuple = [(date.year, date.month) for date in dates]
            accumulated = self.transform_presence_into_hour_heatmap(
                species_presence, hours, accumulator=month_tuple
            )
        return accumulated

    @staticmethod
    def transform_presence_into_hour_heatmap(
        species_presence, hours, accumulator
    ):
        """
        Transform per-embedding presence into a 24-hour by time-bin matrix.

        Parameters
        ----------
        species_presence : np.ndarray
            binary presence per embedding
        hours : np.ndarray
            hour of day for each embedding
        accumulator : list of tuples
            time bin label (e.g., (year, month, day)) for each embedding

        Returns
        -------
        np.ndarray
            heatmap of shape (24, n_time_bins) with -1 for empty bins
        """
        accumulated = (
            np.ones([24, len(np.unique(accumulator, axis=0))], dtype=np.int8)
            * -1
        )
        for acc_idx, item in enumerate(np.unique(accumulator, axis=0)):
            month_presence_idx = np.where(np.all(accumulator == item, axis=1))[
                0
            ]
            for hour in range(24):
                hourly_presence_idx = np.where(
                    hours[month_presence_idx] == hour
                )[0]
                if len(hourly_presence_idx) > 0:
                    try:
                        accumulated[hour, acc_idx] = sum(
                            species_presence[
                                month_presence_idx[hourly_presence_idx]
                            ]
                        )
                    except Exception as e:
                        raise Exception
        return accumulated

    def get_timestamps_per_embedding(self, model):
        """
        Compute a datetime timestamp for each embedding.

        Parameters
        ----------
        model : str
            name of the model

        Returns
        -------
        None
            timestamps are stored on the instance
        """
        from bacpipe.embedding_evaluation.label_embeddings import (
            get_dt_filename,
        )
        import datetime as dt

        # embed_dict = self.vis_loader.embeds[model]
        ts_within_audio_files = [
            dt.timedelta(seconds=ts) for ts in self.embed_dict["timestamp"]
        ]
        unique_audio_files = list(
            set(self.embed_dict["metadata"]["audio_files"])
        )
        unique_audio_files.sort()
        ts_files = [get_dt_filename(f) for f in unique_audio_files]
        ts_files_same_length_as_embeds = []
        [
            ts_files_same_length_as_embeds.extend([ts_file] * embed_len)
            for ts_file, embed_len in zip(
                ts_files, self.embed_dict["metadata"]["nr_embeds_per_file"]
            )
        ]

        self.timestamps = [
            ts_file + ts_within_audio_file
            for ts_file, ts_within_audio_file in zip(
                ts_files_same_length_as_embeds, ts_within_audio_files
            )
        ]
