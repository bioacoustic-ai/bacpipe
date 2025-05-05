# EVALUATION

### Evaluation embeddings using
 - [visual analysis of dimensionality reduction](#visual-analysis-of-dimensionality-reduction)
 - [clustering](#clustering)
 - [classification using linear probing](#classification-using-linear-probing)


## visual analysis of dimensionality reduction

In the config file under `available_dim_reduction_models` you will find all models that are available to reduce the dimensionality of your embeddings. By default the output dimension will be `2` to ensure that the results can be visualized. If a model is selected, once the embeddings are produced, a dimensionality reduction will be performed for all of the generated embeddings and saved in `bacpipe/evaluation/dim_reduced_embeddings`. More specifics on how the data is saved can be found [here](dim_reduced_embeddings/README.md).

If multiple embedding models have been selected, a comparison plot will be generated and saved in the directory corresponding to the last saved reduced dimensions (i.e. in `bacpipe/evaluation/dim_reduced_embeddings`). 

## clustering

When computing a dimensionality reduction, clustering metrics (based on the reduced dimensions) will be saved alongside the visualizations.

## classification using linear probing


Models' performance can also be measured on predefined 
audio tasks listed in `available_evaluation_tasks`. A linear classifier will be trained based on the task and evaluated on classification. (This does not require a GPU, training only takes seconds, as only a single layer is trained). 

**At this point and time publication of the training data that we use is still pending, however you can define your own classification task if you have labeled data.**

**Linear classifiers** are simple linear layers trained for the specific task at hand, until convergence. These are intended to produce predictions on the test set with minimum modification of the pre-trained embeddings. From the predictions generated we compute micro and macro accuracies which provide a comparable value to evaluate performance across the different models. 

### How to use it

Steps:
   1) change in the path in `config.yaml` to your audio data path
   2) have a look at the [config.json](tasks/species/config.json) file to see the structure of the file and the required keywords
   3) have a look at the [dataset_9species.csv](datasets/benchmark/dataset_9species.csv) file to see the structure of the csv file and required columns
      - the main thing that is important here, is to have a column name specifying the labels and in your `config.json` file having a keyword `label_type` specifying that column name
   4) then finally
      - once your audio data is setup
      - you have your `config.json` file in a subdirectory of the `tasks` folder 
      - a `data.csv` file specifying the labels corresponding to the filename
      - you have listed that `data.csv` file path in the `dataset_csv_path` keyword in the `config.json` file
      - you're ready to go 
   5) finally, generate embeddings for the models you would like to evaluate 
      - (You can also generate your embeddings first without a task and the embeddings will be found and reused for the evaluation using the classification)


### Results 

Once the evaluation finished, a `classification_results.json` file will be saved in `evaluations/metrics`. More infor on that [here](evaluations/metrics/README.md).

**Clustering-based evaluation** is based on computing various clustering scores for the embeddings and labels of each task. 
These are calculated directly on the extracted embeddings without any fine tuning for the specific task. 
Some pretrained models are naturally better at certain tasks given their training domain. For instance birdnet on species classification. however these scores are a good indicator of performance on other tasks.

Alongside the metrics, a visualization will be saved showing the performance reults in `evaluations/plots`. More infor on that [here](evaluations/plots/README.md).

### Structure of code:
   * `classification_utils/` : contains helper functions to run the classification steps.
   * `classification.py`, `clustering.py` and `visualisation.py` : main scripts to run specific classification, clustering and visualization steps.
   * `datasets/` : where the audiio data is stored.
   * `dim_reduced_embeddings/` : here the embeddings after the dimensionality reduction are saved
   * `embeddings/` : here all of the embeddings that are computed during model inference are saved 
   * `evaluations/` : here the classificaion results from the tasks are saved
   * `tasks/` : contains the task definitions and task specific parameters
    



