import yaml

import logging
import numpy as np
from bacpipe.generate_embeddings import generate_embeddings
from bacpipe.visualize_umaps import plot_embeddings

logger = logging.getLogger("bacpipe")




def get_embeddings(model_name, audio_dir,
    check_if_primary_combination_exists=False,
    check_if_secondary_combination_exists=False,
):

    generate_embeddings(
        model_name=model_name,
        audio_dir=audio_dir,
        check_if_combination_exists=check_if_primary_combination_exists,
    )
    ld = generate_embeddings(
        model_name="umap",
        audio_dir=audio_dir,
        check_if_combination_exists=check_if_secondary_combination_exists,
    )
    plot_embeddings(ld.embed_dir)
    embeds, divisions_array = [], []
    for ind, file in enumerate(ld.files):
        d = json.load(open(file))
        arr = np.array([d["x"], d["x"]]).reshape([len(d["x"]), 2])
        embeds.append(arr)
        # append_timeList(ld.metadata_dict, ind, divisions_array)

    return embeds, ld.metadata_dict, divisions_array
