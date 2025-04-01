import time
import logging
from pathlib import Path

from tqdm import tqdm
import numpy as np

import bacpipe.generate_embeddings as ge
from bacpipe.embedding_evaluation.visualization.visualize import plot_embeddings

from bacpipe.embedding_evaluation.visualization.visualize import (
    plot_comparison,
    visualize_task_results,
    visualise_classification_results_across_models,
)
import yaml
from bacpipe.embedding_evaluation.classification.main_classify import evaluate_on_task
from bacpipe.embedding_evaluation.classification.evaluate_classifcation_results import (
    build_results_report,
)


logger = logging.getLogger("bacpipe")


with open("config.yaml", "rb") as f:
    config = yaml.load(f, Loader=yaml.CLoader)

with open("bacpipe/path_settings.yaml", "rb") as p:
    paths = yaml.load(p, Loader=yaml.CLoader)


def get_model_names():
    if config["embedding_model"]["already_computed"]:
        dataset_name = Path(config["audio_dir"]).stem
        main_results_path = Path(paths["embed_parent_dir"].format(dataset_name))
        model_names = [
            d.stem.split("___")[-1].split("-")[0]
            for d in list(main_results_path.rglob("*"))
            if d.is_dir()
        ]
    else:
        model_names = config["embedding_model"]["selected_models"]


def model_specific_embedding_creation():
    for model_name in config["embedding_model"]:
        get_embeddings(
            model_name=model_name,
            dim_reduction_model=config["dim_reduction_model"],
            audio_dir=config["audio_dir"],
        )


def model_specific_evaluation():
    for model_name in config["embedding_model"]:
        loader_obj = ge.Loader(
            config["audio_dir"],
            model_name=model_name,
            dim_reduction_model=config["dim_reduction_model"],
        )
        if not config["evaluation_task"] == "None":
            task_name = config["evaluation_task"]
            print(
                "\n#### Training linear probe to evaluate embeddings on the "
                f"classification task {task_name.upper()}. ####"
            )
            assert len(loader_obj.files) > 1, (
                "Too few files to evaluate embeddings with linear probe. "
                + "Are you sure you have selected the right data?"
            )
            metrics, task_config = evaluate_on_task(task_name, model_name, loader_obj)
            if not task_config:
                return None

            build_results_report(task_name, model_name, metrics, task_config)
            visualize_task_results(task_name, model_name, metrics)


def cross_model_evaluation():
    if len(config["embedding_model"]) > 1:
        if not config["evaluation_task"] == "None":
            visualise_classification_results_across_models(
                config["evaluation_task"], config["embedding_model"]
            )
        if not config["dim_reduction_model"] == "None":
            plot_comparison(
                config["audio_dir"],
                config["embedding_model"],
                config["dim_reduction_model"],
            )


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
            logger.debug(
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


def generate_embeddings(save_files=True, **kwargs):
    if "dim_reduction_model" in kwargs:
        print(
            f"\n\n\n###### Generating embeddings using {kwargs['dim_reduction_model'].upper()} ######\n"
        )
    elif "model_name" in kwargs:
        print(
            f"\n\n\n###### Generating embeddings using {kwargs['model_name'].upper()} ######\n"
        )
    else:
        raise ValueError("model_name not provided in kwargs.")
    try:
        start = time.time()
        ld = ge.Loader(**kwargs)
        logger.debug(f"Loading the data took {time.time()-start:.2f}s.")
        if not ld.combination_already_exists:
            embed = ge.Embedder(**kwargs)
            for idx, file in enumerate(
                tqdm(ld.files, desc="processing files", position=1, leave=False)
            ):
                if not ld.dim_reduction_model:
                    sample = file
                    embeddings = embed.get_embeddings_from_model(sample)
                    ld.write_audio_file_to_metadata(
                        idx, file, embed, embeddings.shape[-1]
                    )
                    embed.save_embeddings(idx, ld, file, embeddings)
                else:
                    if idx == 0:
                        embeddings = ld.embed_read(idx, file)
                    else:
                        embeddings = np.concatenate(
                            [embeddings, ld.embed_read(idx, file)]
                        )
            if ld.dim_reduction_model:
                dim_reduced_embeddings = embed.get_embeddings_from_model(embeddings)
                embed.save_embeddings(idx, ld, file, dim_reduced_embeddings)
            ld.write_metadata_file()
            ld.update_files()
        return ld
    except KeyboardInterrupt:
        if ld.embed_dir.exists() and ld.rm_embedding_on_keyboard_interrupt:
            print("KeyboardInterrupt: Exiting and deleting created embeddings.")
            import shutil

            shutil.rmtree(ld.embed_dir)
        import sys

        sys.exit()
