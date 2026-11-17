"""
Runs the bacpipe pipeline in its own process, equivalent to
``python run_pipeline.py``:

    python worker.py path/to/run.yaml

Instead of opening the dashboard, the arguments for
``visualize_using_dashboard`` are stored in ``dashboard_kwargs_file`` for
dashboard.py. ``run.yaml`` contains::

    config:   {...}            values for bacpipe.config
    settings: {...}            values for bacpipe.settings
    dashboard_kwargs_file: ... where to store the dashboard arguments

Must not import bacpipe_app.
"""

import os
import pickle
import sys
import time
import traceback
from pathlib import Path

# make the same packages importable as in the application process
for _entry in os.environ.get("BACPIPE_APP_SYS_PATH", "").split(os.pathsep):
    if _entry and _entry not in sys.path:
        sys.path.append(_entry)


#: Class names of exceptions raised when the internet cannot be reached.
NO_CONNECTION_ERRORS = {
    "gaierror",  # socket: the name of the server could not be resolved
    "NameResolutionError",
    "NewConnectionError",
    "MaxRetryError",
    "ConnectTimeout",
    "ConnectTimeoutError",
    "ConnectError",
    "ProxyError",
    "LocalEntryNotFoundError",  # huggingface_hub: offline and not in the cache
}


def exception_chain(error):
    """The exception and everything that caused it, outermost first."""
    seen = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def is_no_connection_error(error):
    """True if the error (or one of its causes) is a failed connection."""
    for cause in exception_chain(error):
        kind = type(cause)
        if kind.__name__ in NO_CONNECTION_ERRORS:
            return True
        # requests.exceptions.ConnectionError, not the builtin ConnectionError
        # (that one is also raised for broken pipes between processes)
        if kind.__name__ == "ConnectionError" and kind.__module__.startswith("requests"):
            return True
    return False


def no_connection_message(run, traceback_file):
    """What to show instead of the traceback of a failed download."""
    checkpoints = run.get("settings", {}).get("model_base_path", "the model checkpoint folder")
    return (
        "\nNo internet connection: a model could not be downloaded.\n\n"
        "bacpipe downloads the checkpoint of a model the first time the model "
        "is used, and the download server could not be reached.\n"
        "  * Check that this computer is online. A VPN, firewall or proxy can "
        "also block the download.\n"
        "  * Then play the bacpipe again, models that were already downloaded "
        "are not downloaded again.\n"
        "  * Without internet: copy the checkpoints from another computer "
        f"into\n      {checkpoints}\n\n"
        f"The technical details were saved to {traceback_file}\n"
    )


def main(run_file):
    import yaml

    with open(run_file, encoding="utf-8") as f:
        run = yaml.safe_load(f)

    print("Loading bacpipe and its models, this can take a moment ...\n", flush=True)
    started = time.time()

    import bacpipe
    import bacpipe.core.workflows as workflows

    for key, value in run["config"].items():
        setattr(bacpipe.config, key, value)
    for key, value in run["settings"].items():
        setattr(bacpipe.settings, key, value)

    def remember_dashboard_arguments(**kwargs):
        """Stand-in for ``visualize_using_dashboard``."""
        with open(run["dashboard_kwargs_file"], "wb") as f:
            pickle.dump(kwargs, f)

    workflows.visualize_using_dashboard = remember_dashboard_arguments

    try:
        bacpipe.play(bool_save_logs=True, **run["config"], **run["settings"])
    except Exception as error:
        if not is_no_connection_error(error):
            raise
        traceback_file = Path(run_file).with_name("error_traceback.log")
        traceback_file.write_text(traceback.format_exc(), encoding="utf-8")
        print(no_connection_message(run, traceback_file), flush=True)
        sys.exit(1)

    print(f"\nPipeline finished after {time.time() - started:.0f} s.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
