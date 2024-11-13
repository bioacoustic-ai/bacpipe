import json
import os 

def evaluating_on_task(task_name='ID', pretrained_model='birdnet'):
    ''' 
    trains a linear probe and predict on test set for the given task. 
    arguments: task_name -> string, what is the task to evaluate on
    pretrained_model -> string, pretrained model name from where the embeddings were extracted. (model that is being evaluated)

    
    outputs: predictions on test set, 
            overall and per species evaluation metrics.


    '''


    task_details_dict = json.load(open(os.path.join('bacpipe/bacpipe/evaluation/datasets/benchmark/tasks', task_name+'.json', 'r')))
    embeddings_path = os.path.join('bacpipe/bacpipe/evaluation/dim_reduced_embeddings', pretrained_model)

    



    return
