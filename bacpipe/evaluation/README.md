# Evaluation scripts and data to analyze model performance


Models' performance is measured on 3 predefined audio tasks: animal ID classification, Species classification and Taxon classification. 
These tasks and data to be used are defined in _bacpipe/bacpipe/evaluation/tasks/_

For the evaluation we follow 3 approaches: classification with linear probes, clustering metrics, and visualisation. 

**linear probes** are simple linear layers trained for the specific task at hand, until convergence. These are intended to produce predictions on the test set with minimum modeification of the pre-trained embeddings. from the predictions generated we compute micro and macro accuracies which provide a comparable value to evaluate performance across the different models. 

**clustering-based evaluation** is based on computing various clustering scores for the embeddings and labels of each task. 
these are calculated directly on the extracted embeddings without any fine tuning for the specific task. 
Some pretrained models are naturally better at certain tasks given their training domain. for instance birdnet on species classification. however these scores are a good indicator of performance on other tasks.

**viusalisation** 




## structure:
   * eval.py : main script calling the 3 main evaluation steps on the 3 tasks and defined model. 
   
   * classification.py, clustering_eval.py and visualisation.py : scripts to run specific evaluation steps.
   
   * tasks/ : contains the tasks' definitions and task specific parameters.
   
   * results/ : where results are stored.
   * evaluation_utils/ : contains code to run the evaluation steps.
   * datasets/ : where the audiio data is stored.
   * 
    



## Run evaluation - instructions:

1) download the data
2) extract embeddings
3) review configs <---- to CHECK
4) run eval.py
