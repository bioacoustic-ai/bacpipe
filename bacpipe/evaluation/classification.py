import json
import pandas as pd
import yaml


from .evaluation_utils.embedding_dataloader import EmbeddingTaskLoader
from .evaluation_utils.linear_probe import (
    LinearProbe,
    train_linear_probe,
    inference_with_linear_probe,
)
from .evaluation_utils.evaluation_metrics import compute_metrics, build_results_report

import torch


def evaluating_on_task(
    task_name, model_name, loader_object, task_config_path, device_str
):
    """
    trains a linear probe and predicts on test set for the given task.

    arguments: task_name -> string, what is the task to evaluate on
    pretrained_model -> string, pretrained model name from where the embeddings were extracted. (model that is being evaluated)


    outputs: predictions on test set,
            overall and per class evaluation metrics.

    """

    # read config.json
    config = json.load(open(task_config_path, "r"))
    # pretrained_model = config["pretrained_model_name"]

    # device = torch.device(device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset_path = config["dataset_csv_path"]
    data = pd.read_csv(dataset_path)

    # train linear probe
    train_df = data[data["predefined_set"] == "train"]

    train_data = EmbeddingTaskLoader(
        partition_dataframe=train_df,
        pretrained_model_name=model_name,
        loader_object=loader_object,
        target_labels=config["labels"],
        label2index=config["label_to_index"],
    )
    training_generator = torch.utils.data.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )

    test_df = data[data["predefined_set"] == "test"]
    test_data = EmbeddingTaskLoader(
        partition_dataframe=test_df,
        pretrained_model_name=model_name,
        loader_object=loader_object,
        target_labels=config["labels"],
        label2index=config["label_to_index"],
    )
    test_generator = torch.utils.data.DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )
    embed_size = loader_object.metadata_dict["embedding_size"]

    lp = LinearProbe(in_dim=embed_size, out_dim=config["Num_classes"]).to(device)
    lp = train_linear_probe(lp, training_generator, config, device)

    predictions, gt_indexes, probability_scores = inference_with_linear_probe(
        lp, test_generator, device
    )
    # compute the evaluation metrics

    overall_metrics, per_class_metrics = compute_metrics(
        predictions, gt_indexes, probability_scores, config["label_to_index"]
    )

    return predictions, overall_metrics, per_class_metrics


if __name__ == "__main__":
    # example usage
    task_name = "ID"  # TODO: remove from evaluation function arguments?
    pretrained_model = "birdnet"
    embeddings_size = 1024  # TODO:  remove from function arguments, should be read from the model specific configs
    device = "cuda:0"  # TODO: remove from function arguments?

    task_config_path = "/homes/in304/Pretrained-embeddings-for-Bioacoustics/bacpipe/bacpipe/evaluation/tasks/ID/ID.json"
    # TODO embeddings_path = os.path.join('/homes/in304/Pretrained-embeddings-for-Bioacoustics/bacpipe/bacpipe/evaluation/embeddings', pretrained_model)
    embeddings_path = (
        "/import/c4dm-datasets-ext/animal_id_FEBdataset/pretrained_embeddings/birdnet"
    )

    predictions, overall_metrics, per_class_metrics = evaluating_on_task(
        task_name,
        pretrained_model,
        embeddings_size,
        embeddings_path,
        task_config_path,
        device,
    )
    print(predictions)
    print(overall_metrics)
    print(per_class_metrics)

    build_results_report(
        task_name, pretrained_model, overall_metrics, per_class_metrics
    )
