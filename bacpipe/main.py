import logging
import numpy as np
from bacpipe.generate_embeddings import generate_embeddings
from bacpipe.visualize import plot_embeddings

logger = logging.getLogger("bacpipe")


def get_embeddings(
    model_name,
    audio_dir,
    dim_reduction_model="None",
    check_if_primary_combination_exists=True,
    check_if_secondary_combination_exists=False,
):
    generate_embeddings(
        model_name=model_name,
        audio_dir=audio_dir,
        check_if_combination_exists=check_if_primary_combination_exists,
    )
    if dim_reduction_model != "None":
        ld = generate_embeddings(
            model_name=model_name,
            dim_reduction_model=dim_reduction_model,
            audio_dir=audio_dir,
            check_if_combination_exists=check_if_secondary_combination_exists,
        )
        plot_embeddings(ld.embed_dir)
