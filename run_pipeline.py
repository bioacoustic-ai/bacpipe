from bacpipe.main import get_embeddings
from bacpipe.evaluation.visualization import (
    plot_comparison,
    visualize_task_results,
    visualise_classification_results_across_models,
)
import yaml
from bacpipe.evaluation.classification import evaluate_on_task
from bacpipe.evaluation.classification_utils.evaluation_metrics import (
    build_results_report,
)

with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)

overall_macro_acc_models_dict = {}
per_class_acc_across_models_dict = {}


for model_name in config["embedding_model"]:
    loader_obj = get_embeddings(
        model_name=model_name,
        dim_reduction_model=config["dim_reduction_model"],
        audio_dir=config["audio_dir"],
    )
    if not config["evaluation_task"] == "None":
        task_name = config["evaluation_task"]
        print(
            "\n#### Training linear probe to evaluate embeddings on the "
            f"classification task {task_name.upper()}. ####"
        )
        assert len(loader_obj.files) > 1, (
            "Too few files to evaluate embeddings with linear probe. "
            + "Are you sure you have selected the right data?"
        )

        metrics, task_config = evaluate_on_task(task_name, model_name, loader_obj)

        build_results_report(task_name, model_name, metrics, task_config)
        visualize_task_results(task_name, model_name, metrics)
if len(config["embedding_model"]) > 1:
    if not config["evaluation_task"] == "None":
        visualise_classification_results_across_models(
            task_name, config["embedding_model"]
        )
    if not config["dim_reduction_model"] == "None":
        plot_comparison(
            config["audio_dir"],
            config["embedding_model"],
            config["dim_reduction_model"],
        )
