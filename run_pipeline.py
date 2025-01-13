from bacpipe.main import get_embeddings
from bacpipe.evaluation.visualization import plot_comparison, visualize_task_results
import yaml
from bacpipe.evaluation.classification import evaluate_on_task
from bacpipe.evaluation.evaluation_utils.evaluation_metrics import build_results_report

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

        overall_metrics, per_class_metrics, items_per_class = evaluate_on_task(
            task_name, model_name, loader_obj
        )

        build_results_report(
            task_name, model_name, overall_metrics, per_class_metrics, items_per_class
        )
        visualize_task_results(
            task_name, model_name, overall_metrics, per_class_metrics
        )
    if not config["dim_reduction_model"] == "None":
        plot_comparison(
            config["audio_dir"],
            config["embedding_model"],
            config["dim_reduction_model"],
        )
