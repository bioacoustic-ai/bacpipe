from pathlib import Path
import pytest
import bacpipe


def pytest_addoption(parser):
    parser.addoption(
        "--models",
        action="store",
        default=None,
        help="Comma-separated list of models to test (default: all available models)",
    )


def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        option = metafunc.config.getoption("models")

        if option:
            # User-specified models
            models = option.split(",")
        else:
            # Discover all models
            models = [
                mod.stem
                for mod in Path(
                    bacpipe.PACKAGE_MAIN
                    / "embedding_generation_pipelines/feature_extractors"
                ).glob("*.py")
            ]

        if not models:
            models = ["birdnet"]  # fallback if nothing found

        # models_requiring_checkpoints = [
        #     "audiomae",
        #     "aves_especies",
        #     "avesecho_passt",
        #     "beats",
        #     "birdaves_especies",
        #     "hbdet",
        #     "insect66",
        #     "insect459",
        #     "mix2",
        #     "naturebeats",
        #     "protoclr",
        #     "vggish",
        # ]
        # for model in models_requiring_checkpoints:
        #     if (
        #         not Path(f"bacpipe/model_checkpoints/{model}").exists()
        #         and model in models
        #     ):
        #         models.remove(model)

        print(">>> Models selected for tests:", models)
        metafunc.parametrize("model", models)
