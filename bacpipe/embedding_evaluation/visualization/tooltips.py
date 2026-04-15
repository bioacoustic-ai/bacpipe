clustering = """
The clustering evaluation method proposed in bacpipe are external 
evaluation methods (https://doi.org/10.1016/j.neucom.2024.128198), 
meaning that they are based on ground truth. 
However, in the absence of other ground truth, 
validation's measurements like 
AMI (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html) 
will use metadata, such as file_directory and time_of_day, as ground truth. 
This shows if embeddings are structured according to metadata.

For example :
A set of recordings has been analysed through bacpipe. 
When looking at the AMI of clustered embeddings, we can examine the mutual 
information shared between generated clusterings and time_of_the_day, which is 
in the metadata. If the embeddings are structured according to time_of_day, 
most information will be shared between time_of_day and cluster label, resulting 
in an AMI close to 1. If the AMI is close to 0, then the embeddings are not 
structured according to this metadata. We suggest that this should stay an 
exploratory method as this is a fast and effective way to explore relations 
between metadata and embedding's structure. However, appropriate steps should 
be taken for further analysis into the strucutre.
"""

probing = """
The probing evaluation method requires ground truth annotations to be 
present in a specific form: a .csv file saved in the root directory 
of your audio files with the columns: `audiofilename`, `start`, `end`, 
`label:species` (also specified in the ReadMe file). That data is separated
into train, validation and test and using that a linear and knn probe is 
trained and fitted. The results below show the classwise performance 
of the probe.
"""

