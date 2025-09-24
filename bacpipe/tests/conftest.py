from pathlib import Path
from bacpipe import supported_models, models_needing_checkpoint, settings

# cache so we only compute once
_filtered_models = None


def pytest_addoption(parser):
    parser.addoption(
        "--models",
        action="store",
        default=None,
        help="Comma-separated list of models to test (default: all available models)",
    )
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run the tests on (e.g., 'cpu', 'cuda')",
    )

def pytest_generate_tests(metafunc):
    global _filtered_models

    if "model" not in metafunc.fixturenames:
        return

    if _filtered_models is None:
        option = metafunc.config.getoption("models")

        if option:
            # user-specified models
            models = option.split(",")
        else:
            models = list(supported_models)
            not_available = [
                m for m in models_needing_checkpoint
                if not (Path(settings.model_base_path) / m).exists()
            ]
            for m in not_available:
                if m in models:
                    models.remove(m)

            if not models:
                models = ["birdnet"]  # fallback if nothing left

        _filtered_models = models

    print(">>> Models selected for tests:", _filtered_models)
    metafunc.parametrize("model", _filtered_models)

    if "device" in metafunc.fixturenames:
        device_option = metafunc.config.getoption("device")
        metafunc.parametrize("device", [device_option])