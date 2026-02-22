import panel as pn
import matplotlib
import sys
import seaborn as sns
import numpy as np
import logging

logger = logging.getLogger("bacpipe")

import importlib.resources as pkg_resources
import bacpipe.imgs
from .visualize_embeddings import (
    plot_embeddings,
    plot_comparison,
    EmbedAndLabelLoader,
)
from .visualize import (
    plot_clusterings,
    clustering_overview,
    plot_overview_metrics,
)
from .visualize_spectrograms import SpectrogramPlot
from .visualize_predictions import (
    plot_classification_results, 
    plot_classification_heatmap, 
    PredictionsLoader
    )
    
import bacpipe.embedding_evaluation.label_embeddings as le
from bacpipe.embedding_evaluation.classification.train_classifier import LinearClassifier

sns.set_theme(style="whitegrid")

matplotlib.use("agg")

# Enable Panel
pn.extension('plotly')
from .dashboard_utils import DashBoardHelper

class DashBoard(DashBoardHelper):
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
        self.audio_dir = audio_dir
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
        
        self.interactive_embedding_plot = True

        self.model_select = dict()
        self.label_select = dict()
        self.noise_select = dict()
        self.autoplay_audio_select = dict()
        self.clfier_select = dict()
        self.species_select = dict()
        self.accumulate_select = dict()
        self.class_select = dict()
        self.embed_plot = dict()
        
        self.interactive_embed_plot = dict()
        self.spectrogram_plot_panel = dict()
        self.spec_plot_obj = dict() 
        self._trigger_spec_obj_update = dict()
        
        self.class_options = dict()
        self.preds_data = dict()
        self.clfier_path = dict()
        self.clfier_thresh = dict()
        self.btn_run_clfier = dict()
        self.progress_bar = dict()
        self.trigger_classification = dict()
        self.loading_test_placeholder = dict()
        
        self.heatmap_plot = dict()
        self.kwargs = kwargs

    def embedding_panel(self, widget_idx=0):
        if not self.interactive_embedding_plot:
            embedding_plot = self.init_plot(
                            # self.init_interactive_plot(
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
                            )
        else:
            
            self.interactive_embed_plot[widget_idx] = pn.pane.Plotly(
                config={'responsive': True},
                sizing_mode='stretch_both'
            )
            self.interactive_embed_plot[widget_idx].param.watch(
                lambda x: self.handle_click(x, widget_idx), 'click_data'
                )
            self.interactive_embed_plot[widget_idx].param.watch(
                lambda x: self.handle_selection(x, widget_idx), 'selected_data'
                )           
             
            embedding_info_dialogue = pn.widgets.StaticText(
                    value="", width=400
                    )
            
            self.spec_plot_obj[widget_idx] = SpectrogramPlot(
                self.audio_dir, self.vis_loader, 
                self.model_select[widget_idx], 
                embedding_info_dialogue,
                **self.kwargs
                )
            
            self._trigger_spec_obj_update[widget_idx] = pn.bind(
                (
                    self.spec_plot_obj[widget_idx]._update_spec_obj
                ),
                self.model_select[widget_idx],
                self.autoplay_audio_select[widget_idx]
            )
            
            self.spectrogram_plot_panel[widget_idx] = pn.pane.Plotly(
                    SpectrogramPlot.dummy_image(title=''), 
                    sizing_mode='stretch_width',
                    height=300
                )
            
            embedding_plot = pn.bind(
                        self.update_main_plot,
                        # self.init_plot,
                        "interactive_embed",
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
                        )
            
            play_audio_button = pn.widgets.Button(name="Play audio", button_type="primary")
            play_audio_button.on_click(self.spec_plot_obj[widget_idx].play_audio)
            save_selection_dialogue = pn.widgets.StaticText(
                    value="", width=400
                    )
            
            save_selection_button = pn.widgets.Button(
                name="Save selection to file", button_type="primary"
                )
            save_selection_button.on_click(
                lambda x: self.save_selected_points(
                    x, save_selection_dialogue, widget_idx
                    )
                )
            save_selection_dialogue.visible = False
            
            
        return pn.Column(
            embedding_plot,
            embedding_info_dialogue,
            self.spectrogram_plot_panel[widget_idx],
            save_selection_dialogue,
            pn.Row(
                play_audio_button,
                save_selection_button
            ),
            pn.widgets.StaticText(value="", height=80)
        )
    


    def single_model_page(self, widget_idx):
        sidebar = self.make_sidebar(widget_idx, model=True)
    
        main_content = pn.Column(
            pn.pane.Markdown(f"## Model Dashboard"),
            pn.Accordion(
                (
                    "2D Embedding Plot",
                    self.embedding_panel(widget_idx)
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
        sidebar = self.make_sidebar(widget_idx, model=False, all_models=True)

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
    
    def apply_clfier_page(self, widget_idx):
        self.class_options[widget_idx] = []
        sidebar = self.make_sidebar(widget_idx, model=True, classifier_page=True)
        
        # input box where i can input the path to the linear classifier
        self.clfier_path[widget_idx] = pn.widgets.TextInput(
                name='Path to Linear Classifier', 
                placeholder=(
                    self.path_func(self.models[0]).class_path / 'linear_classifier.pt'
                    ).as_posix(),
                width=800,
                max_length=800,
                visible=False
                )
        
        self.clfier_thresh[widget_idx] = pn.widgets.TextInput(
            name='Threshold for classification', 
            placeholder='0.5',
            width=80,
            )
        
        from src.btn_icon import icon_str
        self.btn_run_clfier[widget_idx] = pn.widgets.Button(
            # name='Apply linear classifier', 
            name='Load predictions from integrated classifier', 
            icon=icon_str, 
            width=100, 
            height=30,
            )

        
        self.progress_bar[widget_idx] = pn.indicators.Progress(
            value=0,
            max=100,
            bar_color='primary',
            width=800
        )
        
        self.loading_test_placeholder[widget_idx] = pn.widgets.StaticText(
            name='Preparing classification', 
            value=''
            )
            
        self.clfier_select[widget_idx].param.watch(
            lambda x: self.change_input_options(x, widget_idx=widget_idx),
            'value'
        )
        
        
        self.preds_data[widget_idx] = PredictionsLoader(
            self.vis_loader,
            self.path_func,
            self.models,
            panel_selection=self.species_select[widget_idx],
            progress_bar=self.progress_bar[widget_idx],
            loading_pane=self.loading_test_placeholder[widget_idx]
            )
        
        self.btn_run_clfier[widget_idx].on_click(
            lambda x: self.update_main_plot(
                "heatmap",
                plot_classification_heatmap,
                widget_idx=widget_idx,
                event=x,
                predictions_loader=self.preds_data[widget_idx],
                model=self.model_select[widget_idx], 
                accumulate_by=self.accumulate_select[widget_idx], 
                species=self.species_select[widget_idx],
                threshold=self.clfier_thresh[widget_idx],
                clfier_path=self.clfier_path[widget_idx],
                clfier_type=self.clfier_select[widget_idx]
            )
        )
        
        
        main_content = pn.Column(
            pn.pane.Markdown("## All Models Dashboard"),
            pn.Accordion(
                (
                    "Classification settings",
                    pn.Column(
                        # trigger_input_options,
                        self.clfier_path[widget_idx],
                
                        # after that show me the classes that this 
                        # linear classifier will classify
                        pn.widgets.StaticText(
                            name='Classes', 
                            value=pn.bind(
                                self.preds_data[widget_idx].get_classes, 
                                self.clfier_path[widget_idx]
                                )
                            ),
                
                        # input section to give a threshold for classification
                        self.clfier_thresh[widget_idx],
                    
                        # button to click run
                        self.btn_run_clfier[widget_idx],
                        
                        # placeholder textbox to show that something 
                        # is happening while waiting on embeddings to load
                        self.loading_test_placeholder[widget_idx],
                                    
                        # progbar
                        self.progress_bar[widget_idx],
                    )
                ),
                (
                    "Classification heatmap",
                    self.init_plot(
                        "heatmap",
                        plot_classification_heatmap, 
                        widget_idx=widget_idx,
                        event=None,
                        predictions_loader=self.preds_data[widget_idx],
                        model=self.model_select[widget_idx], 
                        accumulate_by=self.accumulate_select[widget_idx], 
                        species=self.species_select[widget_idx],
                        threshold=self.clfier_thresh[widget_idx]
                        )
                    
                ),
                active=[0, 1, 2],
                # by default create all annotations as one big annotations file
                # # add button to save as raven annotations
                ),
                width=900
            )
        return pn.Row(sidebar, main_content)  # , sizing_mode="stretch_both")
        

    def make_sidebar(
        self, widget_idx, model=True, classifier_page=False, all_models=False
        ):
        widgets = [pn.pane.Markdown("## Settings")]

        if model:
            widgets.append(
                self.init_widget(widget_idx, "model", name="Model", options=self.models)
            )

        if not classifier_page:
            widgets.extend(
                [
                    self.init_widget(
                        widget_idx, "label", name="Label by", options=self.label_by
                    ),
                    (
                        pn.widgets.StaticText(name="", value="View only annotated?")
                        if not self.ground_truth is None
                        else None
                    ),
                    (
                        self.init_widget(
                            widget_idx,
                            "noise",
                            name="remove_noise",
                            options=[True, False],
                            attr="RadioBoxGroup",
                            value=False,
                        )
                        if not self.ground_truth is None
                        else None
                    ),
                    (
                        pn.widgets.StaticText(name="", value="Autoplay audio?")
                        if not (
                            self.interactive_embedding_plot is None
                            or all_models is True
                            )
                        else None
                    ),
                    (
                        self.init_widget(
                            widget_idx,
                            "autoplay_audio",
                            name="Autoplay audio",
                            options=[True, False],
                            attr="RadioBoxGroup",
                            value=False,
                        )
                        if not (
                            self.interactive_embedding_plot is None
                            or all_models is True
                            )
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
        else:
            widgets.extend(
                [
                    self.init_widget(
                        widget_idx,
                        w_type="clfier",
                        name="Integrated or linear classifier",
                        options=['Integrated', 'Linear'],
                        attr="RadioBoxGroup",
                        value='Integrated',
                        ),
                    self.init_widget(
                        widget_idx, 
                        w_type='species', 
                        name='Select species', 
                        options=self.class_options[widget_idx]
                        ),
                    self.init_widget(
                        widget_idx, 
                        w_type='accumulate', 
                        name='Select what to aggregate by', 
                        options=['day', 'week', 'month']
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
        apply_classifier0_page = self.apply_clfier_page(0)
        apply_classifier1_page = self.apply_clfier_page(1)

        # Extract sidebars and content
        sidebar0, content0 = model0_page.objects
        sidebar1, content1 = model1_page.objects
        sidebar2, content2 = apply_classifier0_page.objects
        sidebar3, content3 = apply_classifier1_page.objects

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
            ("Apply Classifer", apply_classifier1_page),
            (
                "Two classifiers",
                pn.Row(
                    pn.Column(sidebar2, sidebar3),
                    pn.Row(content2, content3),
                    sizing_mode="stretch_both",
                ),
            ),
            
        )
        
        self.add_styling(model1_page, model_all_page, apply_classifier1_page)
        
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
                logger.info("Shutting down dashboard server...")
                sys.exit(0)

            close_button.on_click(shutdown_callback)

            sidebar.append(close_button)

def visualize_using_dashboard(models, **kwargs):
    """
    Create and serve the dashboard for visualization.

    Parameters
    ----------
    models : list
        embedding models
    kwargs : dict
        Dictionary with parameters for dashboard creation
    """
    from bacpipe.embedding_evaluation.visualization.dashboard import DashBoard
    import panel as pn

    # Configure dashboard
    dashboard = DashBoard(models, **kwargs)

    # Build the dashboard layout
    try:
        dashboard.build_layout()
    except Exception as e:
        logger.exception(
            f"\nError building dashboard layout: {e}\n \n "
            "Are you sure all the evaluations have been performed? "
            "If not, rerun the pipeline with `overwrite=True`.\n \n "
        )
        raise e

    with pkg_resources.path(bacpipe.imgs, 'bacpipe_favicon_white.png') as p:
        favicon_path = str(p)

    template = pn.template.BootstrapTemplate(
        site="bacpipe dashboard",
        title="Explore embeddings of audio data",
        favicon=str(favicon_path),  # must be a path ending in .ico, .png, etc.
        main=[dashboard.app],
    )
    

    template.show(port=5006, address="localhost")
