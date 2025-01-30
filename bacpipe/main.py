import logging
from bacpipe.generate_embeddings import generate_embeddings
from bacpipe.evaluation.visualization import plot_embeddings

logger = logging.getLogger("bacpipe")


def get_embeddings(
    model_name,
    audio_dir,
    dim_reduction_model="None",
    check_if_primary_combination_exists=True,
    check_if_secondary_combination_exists=True,
):
    loader_embeddings = generate_embeddings(
        model_name=model_name,
        audio_dir=audio_dir,
        check_if_combination_exists=check_if_primary_combination_exists,
    )
    if not dim_reduction_model == "None":

        assert len(loader_embeddings.files) > 1, (
            "Too few files to perform dimensionality reduction. "
            + "Are you sure you have selected the right data?"
        )
        loader_dim_reduced = generate_embeddings(
            model_name=model_name,
            dim_reduction_model=dim_reduction_model,
            audio_dir=audio_dir,
            check_if_combination_exists=check_if_secondary_combination_exists,
        )
        if loader_dim_reduced.embed_dir.joinpath("embed.png").exists():
            print(
                f"Embedding visualization already exist in {loader_dim_reduced.embed_dir}"
                " Skipping visualization generation."
            )
        else:
            print(
                "### Generating visualizations of embeddings using "
                f"{dim_reduction_model}. Plots are saved in "
                f"{loader_dim_reduced.embed_dir} ###"
            )
            plot_embeddings(loader_dim_reduced.embed_dir, dim_reduction_model)
    return loader_embeddings
