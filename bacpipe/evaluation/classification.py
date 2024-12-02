import json
import os 

import pandas as pd

from bacpipe.evaluation.evaluation_utils.embedding_dataloader import EmbeddingTaskLoader
from bacpipe.evaluation.evaluation_utils.linear_probe import LinearProbe, train_linear_probe, inference_with_linear_probe
from bacpipe.evaluation.evaluation_utils.evaluation_metrics import compute_metrics

import torch


def evaluating_on_task(task_name='ID',   pretrained_model='birdnet', embeddings_path, task_config_path,):
    ''' 
    trains a linear probe and predicts on test set for the given task. 

    arguments: task_name -> string, what is the task to evaluate on
    pretrained_model -> string, pretrained model name from where the embeddings were extracted. (model that is being evaluated)

    
    outputs: predictions on test set, 
            overall and per class evaluation metrics.

    '''







    

    # read config.json
    config = json.load(open(task_config_path, 'r'))
    pt_model = config["pretrained_model_name"]

    # Move to appropriate device if GPU is available
    device = config["device"]

    # load dataset
    dataset_path = config["dataset_csv_path"]
    data =  pd.read_csv(dataset_path)

    # train linear probe
    train_df = data[data['predefined_set'].isin(config['train_folds'])]

    train_data = EmbeddingTaskLoader(partition_dataframe=train_df, pretrained_model_name=pt_model,
                                                        embeddings_path = embeddings_path,
                                                        target_labels=config['labels'], label2index=config["label_to_index"])
    training_generator = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    test_df = data[data['predefined_set'].isin(config['test_folds'])]
    test_data = EmbeddingTaskLoader(partition_dataframe=test_df, pretrained_model_name=pt_model,
                                                        embeddings_path = embeddings_path,
                                                        target_labels='hierarchical_labels', label2index=config["ids_to_index"])
    test_generator = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    



    lp = LinearProbe(in_dim=config["input_embeddings_size"], out_dim=config["Num_ids"]).to(device)
    lp= train_linear_probe(lp, training_generator, config)

    predictions = inference_with_linear_probe(lp, test_generator, config)
    # compute the evaluation metrics

    overall_metrics, per_species_metrics = compute_metrics(predictions, test_df, config["ids_to_index"])

    
    return predictions, overall_metrics, per_species_metrics
    






if __name__ == '__main__':
    # example usage
    task_name = 'ID'
    pretrained_model = 'birdnet'


    task_config_path = '/homes/in304/Pretrained-embeddings-for-Bioacoustics/bacpipe/bacpipe/evaluation/tasks/ID/ID.json'
    embeddings_path = os.path.join('/homes/in304/Pretrained-embeddings-for-Bioacoustics/bacpipe/bacpipe/evaluation/embeddings', pretrained_model)


    predictions, overall_metrics, per_species_metrics = evaluating_on_task(task_name, pretrained_model, embeddings_path, task_config_path)
    print(predictions)
    print(overall_metrics)
    print(per_species_metrics)