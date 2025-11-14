import panel as pn
import matplotlib
import sys
import json
import seaborn as sns
import numpy as np

import torch.nn.functional as F
import torch

import importlib.resources as pkg_resources
import bacpipe.imgs
from pathlib import Path
from .visualize import (
    plot_embeddings,
    plot_comparison,
    plot_clusterings,
    clustering_overview,
    plot_overview_metrics,
    plot_classification_results,
    EmbedAndLabelLoader,
)
import bacpipe.embedding_evaluation.label_embeddings as le
from bacpipe.embedding_evaluation.classification.train_classifier import LinearClassifier

sns.set_theme(style="whitegrid")

matplotlib.use("agg")

# Enable Panel
pn.extension()


class DashBoard:
    def __init__(
        self,
        model_names,
        audio_dir,
        main_results_dir,
        default_label_keys,
        evaluation_task,
        dim_reduction_model,
        dim_reduc_parent_dir,
        **kwargs,
    ):
        self.models = model_names
        self.default_label_keys = default_label_keys
        self.path_func = le.make_set_paths_func(
            audio_dir, main_results_dir, dim_reduc_parent_dir, **kwargs
        )
        self.label_by = default_label_keys.copy()
        if (
            self.path_func(model_names[0]).class_path
            / "default_classifier_annotations.csv"
        ).exists() and not "default_classifier" in self.label_by:
            self.label_by += ["default_classifier"]
        self.plot_path = self.path_func(model_names[0]).plot_path.parent.parent
        self.dim_reduc_parent_dir = dim_reduc_parent_dir

        self.ground_truth = None
        if (
            le.get_paths(model_names[0])
            .labels_path.joinpath("ground_truth.npy")
            .exists()
        ):
            ground_truth = np.load(
                le.get_paths(model_names[0]).labels_path.joinpath("ground_truth.npy"),
                allow_pickle=True,
            ).item()
            labels = np.unique(
                [lab.split(":")[-1] for lab in ground_truth.keys()]
            ).tolist()
            self.ground_truth = True
            self.label_by += labels

        if len(list(le.get_paths(model_names[0]).clust_path.glob("*.npy"))) > 0:
            self.label_by += ["kmeans"]

        self.evaluation_task = evaluation_task
        self.dim_reduction_model = dim_reduction_model
        self.widget_width = 100
        self.vis_loader = EmbedAndLabelLoader(
            dim_reduction_model=dim_reduction_model,
            default_label_keys=default_label_keys,
            **kwargs,
        )

        self.model_select = dict()
        self.label_select = dict()
        self.noise_select = dict()
        self.class_select = dict()
        self.embed_plot = dict()
        self.kwargs = kwargs

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

    def single_model_page(self, widget_idx):
        sidebar = self.make_sidebar(widget_idx, model=True)

        main_content = pn.Column(
            pn.pane.Markdown(f"## Model Dashboard"),
            pn.Accordion(
                (
                    "2D Embedding Plot",
                    self.init_plot(
                        "embed",
                        plot_embeddings,
                        widget_idx,
                        loader=self.vis_loader,
                        model_name=self.model_select[widget_idx],
                        label_by=self.label_select[widget_idx],
                        ground_truth=self.ground_truth,
                        dim_reduction_model=self.dim_reduction_model,
                        remove_noise=(
                            self.noise_select[widget_idx]
                            if len(self.noise_select.keys()) > 0
                            else False
                        ),
                        dashboard=True,
                        dashboard_idx=widget_idx,
                    ),
                ),
                (
                    "Clustering Results",
                    (
                        self.plot_widget(
                            plot_clusterings,
                            path_func=self.path_func,
                            model_name=self.model_select[widget_idx],
                            label_by=self.label_select[widget_idx],
                            no_noise=(
                                self.noise_select[widget_idx]
                                if len(self.noise_select.keys()) > 0
                                else False
                            ),
                        )
                        if "clustering" in self.evaluation_task
                        else pn.pane.Markdown(
                            "No clustering task specified. "
                            "Please check the config file."
                        )
                    ),
                ),
                (
                    "Classification Performance",
                    (
                        self.plot_widget(
                            plot_classification_results,
                            path_func=self.path_func,
                            task_name=self.class_select[widget_idx],
                            model_name=self.model_select[widget_idx],
                            return_fig=True,
                        )
                        if "classification" in self.evaluation_task
                        else pn.pane.Markdown(
                            "No classification task specified. "
                            "Please check the config file."
                        )
                    ),
                ),
                # sizing_mode="stretch_width",
                active=[0, 1, 2],
            ),
            width=900,
            # sizing_mode="stretch_both",
        )

        return pn.Row(sidebar, main_content)  # , sizing_mode="stretch_both")

    def all_models_page(self, widget_idx):
        sidebar = self.make_sidebar(widget_idx, model=False)

        main_content = pn.Column(
            pn.pane.Markdown("## All Models Dashboard"),
            pn.Accordion(
                (
                    "Embedding Comparison",
                    self.init_plot(
                        "embed",
                        plot_comparison,
                        widget_idx,
                        loader=self.vis_loader,
                        plot_path=self.plot_path,
                        models=self.models,
                        dim_reduction_model=self.dim_reduction_model,
                        label_by=self.label_select[widget_idx],
                        remove_noise=(
                            self.noise_select[widget_idx]
                            if len(self.noise_select.keys()) > 0
                            else False
                        ),
                        default_label_keys=self.default_label_keys,
                        dashboard=True,
                    ),
                ),
                (
                    "Clustering Overview",
                    (
                        self.plot_widget(
                            clustering_overview,
                            path_func=self.path_func,
                            model_list=self.models,
                            label_by=self.label_select[widget_idx],
                            no_noise=(
                                self.noise_select[widget_idx]
                                if len(self.noise_select.keys()) > 0
                                else False
                            ),
                            **self.kwargs
                        )
                        if "clustering" in self.evaluation_task
                        else pn.pane.Markdown(
                            "No clustering task specified. "
                            "Please check the config file."
                        )
                    ),
                ),
                (
                    "Classification Metrics",
                    (
                        self.plot_widget(
                            plot_overview_metrics,
                            plot_path=None,
                            metrics=None,
                            task_name=self.class_select[widget_idx],
                            path_func=self.path_func,
                            model_list=self.models,
                            return_fig=True,
                        )
                        if "classification" in self.evaluation_task
                        else pn.pane.Markdown(
                            "No classification task specified. "
                            "Please check the config file."
                        )
                    ),
                ),
                # sizing_mode="stretch_width",
                active=[0, 1, 2],
            ),
            width=1700,
            # sizing_mode="stretch_both",
        )

        return pn.Row(sidebar, main_content)  # , sizing_mode="stretch_both")
    
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
            probs.append(F.softmax(logits, dim=0).detach().cpu().numpy())
            self.progress_bar.value = int((idx+1)/len(embeds)*100)
        return np.array(probs)
            
    
    def classify_embeddings(self, model, path, threshold, event):
        if path == '':
            path = (
                self.path_func(self.models[0]).class_path / 'linear_classifier.pt'
                ).as_posix()
        
        embeds = self.collect_all_embeddings(model)
        
        with open(Path(path).parent / 'label2index.json', 'r') as f:
            label2index = json.load(f)
        clfier_weights = torch.load(path)
        clfier = LinearClassifier(clfier_weights['clfier.weight'].shape[-1], len(label2index))
        clfier.load_state_dict(clfier_weights)
        clfier.to(self.kwargs['device'])
        
        probs = self.run_classifier(embeds, clfier)
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
            
    def apply_clfier_page(self, widget_idx):
        sidebar = self.make_sidebar(widget_idx, model=True)
        
        clfier_path = pn.widgets.TextInput(
            name='Path to Linear Classifier', 
            placeholder=(
                self.path_func(self.models[0]).class_path / 'linear_classifier.pt'
                ).as_posix(),
            max_length=200
            )
        
        clfier_thresh = pn.widgets.TextInput(
            name='Threshold for classification', 
            placeholder='0.5'
            )
        
        from src.btn_icon import icon_str
        btn_run = pn.widgets.Button(
            name='Apply linear classifier', 
            icon=icon_str, 
            width=100, 
            height=30
            )
        
        self.progress_bar = pn.indicators.Progress(
            value=0,
            max=100,
            bar_color='primary',
        )
        upd_ind = pn.bind(self.classify_embeddings, self.model_select[widget_idx], clfier_path, clfier_thresh)
        
        # connect button click to update function
        prediction_tuple = btn_run.on_click(upd_ind)
                
        main_content = pn.Column(
            # input box where i can input the path to the linear classifier
            clfier_path,
            
            # after that show me the classes that this linear classifier will classify
            pn.widgets.StaticText(name='Classes', value=pn.bind(self.get_classes, clfier_path)),
            
            # input section to give a threshold for classification
            clfier_thresh,
            
            # button to click run
            btn_run,
            
            # progbar
            self.progress_bar,
            
            # heatmap for 20 top species
            pn.bind(self.plot_heatmap, prediction_tuple, self.model_select[widget_idx]),
                    # if len(prediction_tuple) == 2
                    # else None
            
            # by default create all annotations as one big annotations file
            
            # add button to save as raven annotations
        )
        return pn.Row(sidebar, main_content)  # , sizing_mode="stretch_both")
        
    def plot_heatmap(self, prediction_tuple, model):
        # probabilities, class_dict = prediction_tuple
        # timestamps = self.get_timestamps(self.path_func(model).eval_path, model, 'continuous_timestamp')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        sns.heatmap([[1, 2], [2, 1]])      
        return fig      

    def make_sidebar(self, widget_idx, model=True):
        widgets = [pn.pane.Markdown("## Settings")]

        if model:
            widgets.append(
                self.init_widget(widget_idx, "model", name="Model", options=self.models)
            )

        widgets.extend(
            [
                self.init_widget(
                    widget_idx, "label", name="Label by", options=self.label_by
                ),
                (
                    pn.widgets.StaticText(name="", value="Remove noise?")
                    if not self.ground_truth is None
                    else None
                ),
                (
                    self.init_widget(
                        widget_idx,
                        "noise",
                        name="Remove Noise",
                        options=[True, False],
                        attr="RadioBoxGroup",
                        value=False,
                    )
                    if not self.ground_truth is None
                    else None
                ),
                (
                    self.init_widget(
                        widget_idx,
                        "class",
                        name="Classification Type",
                        options=["knn", "linear"],
                    )
                    if "classification" in self.evaluation_task
                    else None
                ),
            ]
        )

        return pn.Column(*widgets, width=200, margin=(10, 10))

    def build_layout(self):
        """
        Builds the layout for the dashboard with two models and a single model page.
        The layout consists of a single model page, a two-models comparison page,
        and a page showing all models. Each page contains sidebars with model-specific
        information and content areas for visualizations.
        """
        
        
        # Build both model pages to initialize widgets
        model0_page = self.single_model_page(0)
        model1_page = self.single_model_page(1)
        model_all_page = self.all_models_page(1)
        apply_classifier_page = self.apply_clfier_page(1)

        # Extract sidebars and content
        sidebar0, content0 = model0_page.objects
        sidebar1, content1 = model1_page.objects

        # Wrap sidebars with titles
        sidebar0 = pn.Column(
            pn.pane.Markdown("## Model 1"), sidebar0  # , sizing_mode="stretch_height"
        )
        sidebar1 = pn.Column(
            pn.pane.Markdown("## Model 2"), sidebar1  # , sizing_mode="stretch_height"
        )

        self.app = pn.Tabs(
            ("Single model", model1_page),
            (
                "Two models",
                pn.Row(
                    pn.Column(sidebar0, sidebar1),
                    pn.Row(content0, content1),
                    sizing_mode="stretch_both",
                ),
            ),
            ("All models", model_all_page),
            ("Apply Classifer", apply_classifier_page)
        )
        
        self.add_styling(model1_page, model_all_page, apply_classifier_page)
        
    def add_styling(self, *pages):
        with pkg_resources.path(bacpipe.imgs, 'bacpipe_unlabelled.png') as p:
            logo_path = str(p)

        for page in pages:
            sidebar = page.objects[0]
            # Add logo to the sidebar
            sidebar.append(
                pn.pane.PNG(logo_path, sizing_mode="scale_width")
            )

            # Add a spacer + contact info below the logo
            sidebar.append(pn.Spacer(height=20))
            sidebar.append(
                pn.pane.Markdown(
                    """
                    **Contact**
                    
                    If you run into problems, please raise issues on github
                    
                    Please collaborate and help make bacpipe as convenient for many as possible
                    
                    🌍 [github](https://github.com/bioacoustic-ai/bacpipe)  
                    
                    To stay updated with new releases, subscribe to the [newsletter](https://buttondown.com/vskode)
                    """
                )
            )
            # Add close button to the header
            close_button = pn.widgets.Button(name="❌ close dashboard")

            def shutdown_callback(event):
                print("Shutting down dashboard server...")
                sys.exit(0)

            close_button.on_click(shutdown_callback)

            sidebar.append(close_button)
        

    def add_save_button(self, plot_func, **kwargs):
        """
        Adds a save button to the plot panel that allows saving the figure
        generated by the provided plotting function. The button will save the
        figure with a filename based on the model name and plot type.

        Parameters
        ----------
        plot_func : function
            The plotting function that generates the figure to be saved.

        Returns
        -------
        pn.Column
            A Panel Column containing the figure panel, a button to save the figure,
            and a notification area to inform the user about the save status.
        """
        # Create the figure panel first using pn.bind
        fig_panel = pn.panel(pn.bind(plot_func, **kwargs))

        # Define the save function that correctly handles panel widget values
        def save_figure(event):
            # Create a copy of kwargs to modify for the direct function call
            plot_kwargs = {}
            for key, value in kwargs.items():
                # Handle panel widgets by getting their value
                if hasattr(value, "value"):
                    plot_kwargs[key] = value.value
                else:
                    plot_kwargs[key] = value

            # Generate the figure with the processed arguments
            fig = plot_func(**plot_kwargs)

            # Generate filename based on processed arguments
            if "model_name" in plot_kwargs:
                model_name = plot_kwargs["model_name"]
            else:
                model_name = "all_models"

            plot_type = plot_func.__name__.replace("plot_", "")
            default_filename = "{}_{}_{}.png".format(
                model_name, plot_type, kwargs["label_by"].value
            )

            # Determine save path
            if model_name == "all_models":
                save_dir = (
                    self.path_func(model_name).plot_path.parent.parent / "overview"
                )
            else:
                save_dir = self.path_func(model_name).plot_path
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / default_filename

            # Save the figure
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

            # Show a notification that saving was successful
            notification.object = f"✓ Figure saved to: {save_path}"

        # Create the button and notification area
        button = pn.widgets.Button(name="Save Figure", button_type="primary")
        button.on_click(save_figure)
        notification = pn.pane.Markdown("")

        # Return the assembled panel
        return pn.Column(fig_panel, pn.Row(button), notification)
