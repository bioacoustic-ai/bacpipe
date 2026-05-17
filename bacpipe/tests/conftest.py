from pathlib import Path
from bacpipe import supported_models, models_needing_checkpoint, settings
import importlib.resources as pkg_resources
import bacpipe
import yaml
import pytest

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
        default=None, #"cpu",
        help="Device to run the tests on (e.g., 'cpu', 'cuda')",
    )
    parser.addoption(
        "--overwrite",
        action="store",
        default=None, #True,
        help="overwrite already processed results or not",
    )
    parser.addoption(
        "--only_embed_annotations",
        action="store",
        default=None, #False,
        help="Only embed annotations or create annotations from grid based on model input length",
    )
    parser.addoption(
        "--check_if_already_processed",
        action="store",
        default=None, #False,
        help="Check if embeddings already exist, if so use existing ones. if False, will overwrite embeddings.",
    )
    parser.addoption(
        "--check_if_already_dim_reduced",
        action="store",
        default=None, #False,
        help="Check if dimensionality reduced embeddings already exist, if so use existing ones. if False, will overwrite.",
    )

# List of boolean options to potentially parametrize
BOOL_OPTIONS = [
    'overwrite',
    'only_embed_annotations', 
    'check_if_already_processed',
    'check_if_already_dim_reduced',
]

def pytest_generate_tests(metafunc):
    global _filtered_models

    # --- models ---
    if "model" in metafunc.fixturenames:
        if _filtered_models is None:
            option = metafunc.config.getoption("models")
            if option == 'tf':
                models = bacpipe.TF_MODELS
            elif option == 'torch':
                all_models = list(supported_models)
                models = [
                    m for m in all_models if m not in bacpipe.TF_MODELS
                    ]
            elif option:
                models = option.split(",")
            else:
                models = bacpipe.TF_MODELS
                    
            if not models:
                models = ["birdnet"]
            
            _filtered_models = models
            
        metafunc.parametrize("model", _filtered_models)


    if "device" in metafunc.fixturenames:
        raw = metafunc.config.getoption("device")
        if raw is None:
            metafunc.parametrize("device", ["cpu", "cuda"])
        else:
            metafunc.parametrize("device", [raw])

    # --- boolean options ---
    for opt in BOOL_OPTIONS:
            if opt in metafunc.fixturenames:
                raw = metafunc.config.getoption(opt)
                if raw is None:
                    vals = [False, True]
                    # This creates labels like 'overwrite=True' and 'overwrite=False'
                    metafunc.parametrize(opt, vals, ids=[f"{opt.split('_')[0]}={str(v)[0]}" for v in vals])
                else:
                    val = False if raw == 'False' else True
                    metafunc.parametrize(opt, [val], ids=[f"{opt.split('_')[0]}={val}"])


@pytest.fixture
def device(request):
    return request.param

@pytest.fixture  
def overwrite(request):
    return request.param

@pytest.fixture
def only_embed_annotations(request):
    return request.param

@pytest.fixture
def check_if_already_processed(request):
    return request.param

@pytest.fixture
def check_if_already_dim_reduced(request):
    return request.param

@pytest.fixture(scope='function')
def kwargs(request):
    # 1. Load your YAMLs once
    with pkg_resources.open_text(bacpipe, "settings.yaml") as f:
        settings_dict = yaml.load(f, Loader=yaml.CLoader)
    with pkg_resources.open_text(bacpipe, "config.yaml") as f:
        config_dict = yaml.load(f, Loader=yaml.CLoader)

    settings_dict["testing"] = True

    # 2. Dynamically pull values ONLY if the test is using those fixtures
    # This prevents unnecessary parametrization for tests that don't list them
    opts = ['device', 'overwrite', 'only_embed_annotations', 
            'check_if_already_processed', 'check_if_already_dim_reduced']
    
    for opt in opts:
        if opt in request.fixturenames:
            # Get the value from the parametrized fixture
            val = request.getfixturevalue(opt)
            settings_dict[opt] = val
        else:
            # Use the default from the YAML or a fallback
            # (Ensures bacpipe.settings remains consistent)
            pass 

    # 3. Sync with global bacpipe.settings
    for k, v in settings_dict.items():
        if hasattr(bacpipe.settings, k):
            setattr(bacpipe.settings, k, v)

    with pkg_resources.path(bacpipe.tests, "test_data") as audio_dir:
        config_dict["audio_dir"] = Path(audio_dir)

    return {**config_dict, **settings_dict}