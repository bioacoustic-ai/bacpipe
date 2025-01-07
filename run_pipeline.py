from bacpipe.main import get_embeddings
from bacpipe.evaluation.visualization import plot_comparison
import yaml
from bacpipe.evaluation.classification import evaluating_on_task, build_results_report

with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)

for model_name in config["embedding_model"]:
    loader_obj = get_embeddings(
        model_name=model_name,
        dim_reduction_model=config["dim_reduction_model"],
        audio_dir=config["audio_dir"],
    )
    if not config["evaluation_task"] == "None":
        # task_name = "ID"  # TODO: remove from evaluation function arguments?
        # pretrained_model = "birdnet"
        # embeddings_size = 1024  # TODO:  remove from function arguments, should be read from the model specific configs
        device = "cuda:0"  # TODO: remove from function arguments?

        task_config_path = "bacpipe/evaluation/tasks/ID/ID.json"
        # TODO embeddings_path = os.path.join('/homes/in304/Pretrained-embeddings-for-Bioacoustics/bacpipe/bacpipe/evaluation/embeddings', pretrained_model)
        # embeddings_path = (
        #     "bacpipe/evaluation/embeddings/"
        # )
        task_name = config["evaluation_task"]

        predictions, overall_metrics, per_class_metrics = evaluating_on_task(
            task_name,
            model_name,
            loader_obj,
            task_config_path,
            device,
        )
        print(predictions)
        print(overall_metrics)
        print(per_class_metrics)

        build_results_report(task_name, model_name, overall_metrics, per_class_metrics)
if not config["dim_reduction_model"] == "None":
    plot_comparison(
        config["audio_dir"], config["embedding_model"], config["dim_reduction_model"]
    )
