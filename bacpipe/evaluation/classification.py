import json
import os 

import pandas as pd

from bacpipe.evaluation.evaluation_utils.embedding_dataloader import EmbeddingTaskLoader
from bacpipe.evaluation.evaluation_utils.linear_probe import LinearProbe, train_linear_probe, inference_with_linear_probe
from bacpipe.evaluation.evaluation_utils.evaluation_metrics import compute_metrics, build_results_report

import torch


def evaluating_on_task(task_name='ID',   pretrained_model='birdnet',embeddings_size=1024,  embeddings_path='bacpipe/bacpipe/evaluation/embeddings', task_config_path='bacpipe/bacpipe/evaluation/tasks/ID/ID.json', device = 'cuda'):
    ''' 
    trains a linear probe and predicts on test set for the given task. 

    arguments: task_name -> string, what is the task to evaluate on
    pretrained_model -> string, pretrained model name from where the embeddings were extracted. (model that is being evaluated)

    
    outputs: predictions on test set, 
            overall and per class evaluation metrics.

    '''


    
    # read config.json
    config = json.load(open(task_config_path, 'r'))
    # pretrained_model = config["pretrained_model_name"]

    # Move to appropriate device if GPU is available
    # device = config["device"]
    

    # load dataset
    dataset_path = config["dataset_csv_path"]
    data =  pd.read_csv(dataset_path)

    # train linear probe
    train_df = data[data['predefined_set']=="train"]

    train_data = EmbeddingTaskLoader(partition_dataframe=train_df, pretrained_model_name=pretrained_model,
                                                        embeddings_path = embeddings_path,
                                                        target_labels=config['labels'], label2index=config["label_to_index"])
    training_generator = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    test_df = data[data['predefined_set']=="test"]
    test_data = EmbeddingTaskLoader(partition_dataframe=test_df, pretrained_model_name=pretrained_model,
                                                        embeddings_path = embeddings_path,
                                                        target_labels=config['labels'], label2index=config["label_to_index"])
    test_generator = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, drop_last=False)


    lp = LinearProbe(in_dim=embeddings_size, out_dim=config["Num_classes"]).to(device)
    lp= train_linear_probe(lp, training_generator, config, device, wandb_configs_path='/homes/in304/Pretrained-embeddings-for-Bioacoustics/bacpipe/bacpipe/evaluation/evaluation_utils/wandb_config.json')

    predictions, gt_indexes = inference_with_linear_probe(lp, test_generator, device)
    # compute the evaluation metrics

    overall_metrics, per_class_metrics = compute_metrics(predictions, gt_indexes,  test_df, config["label_to_index"])

    
    return predictions, overall_metrics, per_class_metrics
    






if __name__ == '__main__':
    # example usage
    task_name = 'ID' #TODO: remove from evaluation function arguments?
    pretrained_model = 'birdnet'
    embeddings_size = 1024  #TODO:  remove from function arguments, should be read from the model specific configs
    device = 'cuda' #TODO: remove from function arguments?

    task_config_path = '/homes/in304/Pretrained-embeddings-for-Bioacoustics/bacpipe/bacpipe/evaluation/tasks/ID/ID.json'
    embeddings_path = os.path.join('/homes/in304/Pretrained-embeddings-for-Bioacoustics/bacpipe/bacpipe/evaluation/embeddings', pretrained_model)


    predictions, overall_metrics, per_class_metrics = evaluating_on_task(task_name, pretrained_model, embeddings_size, embeddings_path, task_config_path, device, )
    print(predictions)
    print(overall_metrics)
    print(per_class_metrics)

    build_results_report(task_name, pretrained_model, overall_metrics, per_class_metrics)