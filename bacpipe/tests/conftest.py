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


        print(">>> Models selected for tests:", models)
        metafunc.parametrize("model", models)
