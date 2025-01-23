from bacpipe.generate_embeddings import Loader
import numpy as np
from pathlib import Path
from bacpipe.evaluation.classification import evaluate_on_task

audio_dir = Path("bacpipe/evaluation/datasets/audio_test_files")
embed_dir = Path("bacpipe/evaluation/datasets/embedding_test_files")

embeddings = {}

tasks = ["ID", "species", "taxon"]


def loader_fn():
    loader = Loader(
        audio_dir=audio_dir,
        dim_reduction_model="umap",
        check_if_combination_exists=False,
        model_name="aves",
        testing=True,
    )
    assert loader.files is not None and len(loader.files) > 0
    return loader


# Define the pytest_generate_tests hook to generate test cases
def pytest_generate_tests(metafunc):
    if "task" in metafunc.fixturenames:
        # Generate test cases based on the test_data list
        metafunc.parametrize("task", tasks)


def test_task_evaluation(task):
    loader = loader_fn()
    # for task in tasks:
    overall_metrics, per_class_metrics, items_per_class = evaluate_on_task(
        task, "avesecho_passt", loader, testing=task
    )


test_task_evaluation("species")
