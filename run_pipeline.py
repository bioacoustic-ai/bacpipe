from bacpipe.main import get_embeddings
from bacpipe.evaluation.visualization import plot_comparison, visualize_task_results
import yaml
from bacpipe.evaluation.classification import evaluate_on_task
from bacpipe.evaluation.classification_utils.evaluation_metrics import build_results_report

with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)

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

        overall_metrics, per_class_metrics, items_per_class = evaluate_on_task(
            task_name, model_name, loader_obj
        )

        build_results_report(
            task_name, model_name, overall_metrics, per_class_metrics, items_per_class
        )
        visualize_task_results(
            task_name, model_name, overall_metrics, per_class_metrics
        )
if not config["dim_reduction_model"] == "None" and len(config["embedding_model"]) > 1:
    plot_comparison(
        config["audio_dir"],
        config["embedding_model"],
        config["dim_reduction_model"],
    )
