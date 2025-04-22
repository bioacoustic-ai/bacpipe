import panel as pn
import matplotlib
import copy
import seaborn as sns
from pathlib import Path
import numpy as np
from functools import reduce

from .visualize import plot_embeddings, plot_comparison, EmbedAndLabelLoader
import bacpipe.embedding_evaluation.label_embeddings as le

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
        self.label_by = default_label_keys.copy()
        get_paths = le.make_set_paths_func(
            audio_dir, main_results_dir, dim_reduc_parent_dir, **kwargs
        )
        self.plot_path = get_paths(model_names[0]).plot_path.parent.parent.joinpath(
            "overview"
        )
        self.dim_reduc_parent_dir = dim_reduc_parent_dir

        self.ground_truth = None
        if (
            le.get_paths(model_names[0])
            .labels_path.joinpath("ground_truth.npy")
            .exists()
        ):
            self.ground_truth = True
            self.label_by += ["ground_truth"]

        if len(list(le.get_paths(model_names[0]).clust_path.glob("*.npy"))) > 0:
            self.label_by += ["kmeans"]

        self.evaluation_task = evaluation_task
        self.dim_reduction_model = dim_reduction_model
        self.widget_width = 120
        self.vis_loader = EmbedAndLabelLoader(
            dashboard=True, dim_reduction_model=dim_reduction_model, **kwargs
        )

        self.model_select = dict()
        self.label_select = dict()
        self.noise_select = dict()
        self.embed_plot = dict()

    def init_plot(self, p_type, plot_func, widget_idx, **kwargs):
        getattr(self, f"{p_type}_plot")[widget_idx] = pn.panel(
            self.plot_widget(plot_func, widget_idx=widget_idx, **kwargs), tight=True
        )
        return getattr(self, f"{p_type}_plot")[widget_idx]

    def plot_widget(self, plot_func, **kwargs):
        return pn.bind(plot_func, **kwargs)

    def widget(self, name, options, attr="Select", width=120, **kwargs):
        return getattr(pn.widgets, attr)(
            name=name, options=options, width=self.widget_width, **kwargs
        )

    def init_widget(self, idx, w_type, **kwargs):
        getattr(self, f"{w_type}_select")[idx] = self.widget(**kwargs)
        return getattr(self, f"{w_type}_select")[idx]

    def single_model_page(self, widget_idx):
        return pn.Column(
            pn.Row(
                self.init_widget(
                    widget_idx, "model", name="Model", options=self.models
                ),
                self.init_widget(
                    widget_idx, "label", name="Label by", options=self.label_by
                ),
                self.init_widget(
                    widget_idx,
                    "noise",
                    name="Noise?",
                    options=[True, False],
                    attr="RadioBoxGroup",
                    value=False,
                ),
            ),
            self.init_plot(
                "embed",
                plot_embeddings,
                widget_idx,
                loader=self.vis_loader,
                model_name=self.model_select[widget_idx],
                label_by=self.label_select[widget_idx],
                ground_truth=self.ground_truth,
                dim_reduction_model=self.dim_reduction_model,
                remove_noise=self.noise_select[widget_idx],
                dashboard=True,
                dashboard_idx=widget_idx,
            ),
            # pn.panel(self.plot_widget(
            #     plot_cluster,
            #     orig_embeds=embed_dict,
            #     model=self.model_select[widget_idx],
            #     label_by=self.label_select[widget_idx],
            #     no_noise=self.noise_select[widget_idx],
            #     tight=True)),
            # pn.panel(self.plot_widget(
            #     plot_class,
            #     orig_embeds=embed_dict,
            #     model=self.model_select[widget_idx],
            #     label_by=self.label_select[widget_idx],
            #     no_noise=self.noise_select[widget_idx],
            #     tight=True))
        )

    def all_models_page(self, widget_idx):
        return pn.Column(
            pn.Row(
                self.init_widget(
                    widget_idx, "label", name="Label by", options=self.label_by
                ),
                self.init_widget(
                    widget_idx,
                    "noise",
                    name="Noise?",
                    options=[True, False],
                    attr="RadioBoxGroup",
                    value=False,
                ),
            ),
            # pn.pane.Matplotlib(pn.bind(
            self.init_plot(
                "embed",
                plot_comparison,
                widget_idx,
                loader=self.vis_loader,
                plot_path=self.plot_path,
                models=self.models,
                dim_reduction_model=self.dim_reduction_model,
                label_by=self.label_select[widget_idx],
                remove_noise=self.noise_select[widget_idx],
                default_label_keys=self.default_label_keys,
                dashboard=True,
                # ), tight=True, dpi=100, height=600, width=800)
            ),
            # pn.panel(self.plot_widget(
            #     plot_cluster,
            #     orig_embeds=embed_dict,
            #     model=self.model_select[widget_idx],
            #     label_by=self.label_select[widget_idx],
            #     no_noise=self.noise_select[widget_idx],
            #     tight=True)),
            # pn.panel(self.plot_widget(
            #     plot_class,
            #     orig_embeds=embed_dict,
            #     model=self.model_select[widget_idx],
            #     label_by=self.label_select[widget_idx],
            #     no_noise=self.noise_select[widget_idx],
            #     tight=True))
        )

    def build_layout(self):
        self.app = pn.Tabs(
            ("Single model", self.single_model_page(1)),
            (
                "Two models",
                pn.Row(self.single_model_page(0), self.single_model_page(1)),
            ),
            (
                "All models",
                self.all_models_page(1),
            ),
        )

    def generate_widgets():
        embed_dict = get_original_embeds()

        ############## LAYOUT ##############

        # model1 =
        reducer1 = pn.widgets.Select(name="Reducer", options=REDUCERS, width=120)
        label_by1 = pn.widgets.Select(
            name="label_by", options=["ground truth", "hdbscan", "kmeans"], width=120
        )
        percentages1 = pn.widgets.RadioBoxGroup(
            name="Show percentages", options=[True, False], value=False
        )
        no_noise1 = pn.widgets.Checkbox(name="Remove noise", value=False)
        select_clustering = pn.widgets.Select(
            name="Reducer", options=METRICS, width=120
        )
        label1 = pn.widgets.Select(
            name="Label",
            options=list(embed_dict[model1.value]["label_dict"].keys()),
            width=120,
        )

        model2 = copy.deepcopy(model1)
        reducer2 = copy.deepcopy(reducer1)
        label_by2 = copy.deepcopy(label_by1)
        percentages2 = copy.deepcopy(percentages1)
        no_noise2 = copy.deepcopy(no_noise1)

        reducer3 = pn.widgets.Select(
            name="Reducer", options=[None] + REDUCERS, width=120
        )
        label3 = pn.widgets.Select(
            name="Label",
            options=list(embed_dict[model1.value]["label_dict"].keys()),
            width=120,
        )
        metric1 = pn.widgets.Select(
            name="Metric", options=["euclidean", "cosine"], width=120
        )

        # Bind the slider to both plots
        Overview = pn.bind(
            plot_overview,
            orig_embeds=embed_dict,
            reducer=reducer1,
            label_by=label_by1,
            no_noise=no_noise1,
        )

        Clusterings_overview = pn.bind(
            plot_clust_overview,
            select_clustering=select_clustering,
            label_by=label_by1,
            no_noise=no_noise1,
            plot_percentages=percentages1,
        )

        Embeddings1 = pn.bind(
            plot_embeds,
            orig_embeds=embed_dict,
            reducer=reducer1,
            model=model1,
            label_keys=list(embed_dict.values())[0]["label_dict"].keys(),
            label_by=label_by1,
            no_noise=no_noise1,
        )
        Embeddings2 = pn.bind(
            plot_embeds,
            orig_embeds=embed_dict,
            reducer=reducer2,
            model=model2,
            label_keys=list(embed_dict.values())[0]["label_dict"].keys(),
            label_by=label_by2,
            no_noise=no_noise2,
            no_legend=True,
        )

        Clusterings1 = pn.bind(
            plot_clusterings,
            model=model1,
            plot_percentages=percentages1,
            no_noise=no_noise1,
        )
        Clusterings2 = pn.bind(
            plot_clusterings,
            model=model2,
            plot_percentages=percentages2,
            no_noise=no_noise2,
            no_legend=True,
        )

        Violins = pn.bind(
            load_distances, reducer=reducer3, model=model1, label=label3, metric=metric1
        )

        Violins2 = pn.bind(load_distances, reducer=reducer1, label=label1)
