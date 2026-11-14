"""
Runs the bacpipe pipeline, started by the application as its own process:

    python worker.py path/to/run.yaml

This is the equivalent of ``python run_pipeline.py``: the values of the form
replace the values of ``config.yaml`` / ``settings.yaml`` and then
``bacpipe.play(bool_save_logs=True)`` is called. Everything the pipeline
prints ends up in the terminal view of the application.

The pipeline runs in its own process so that

* the output (including progress bars and messages of tensorflow/torch) can
  be shown exactly like in a terminal,
* it can be stopped,
* the memory of the models is released when it is finished.

The one difference to ``run_pipeline.py``: the pipeline does not open the
dashboard itself. The arguments it would have used are stored in a file and
the application shows the dashboard inside its own page (see dashboard.py).

``run.yaml`` contains::

    config:   {...}            values for bacpipe.config
    settings: {...}            values for bacpipe.settings
    dashboard_kwargs_file: ... where to store the dashboard arguments

The script must not import bacpipe_app.
"""

import os
import pickle
import sys
import time

# make the same packages importable as in the application process
for _entry in os.environ.get("BACPIPE_APP_SYS_PATH", "").split(os.pathsep):
    if _entry and _entry not in sys.path:
        sys.path.append(_entry)


def main(run_file):
    import yaml

    with open(run_file, encoding="utf-8") as f:
        run = yaml.safe_load(f)

    print("Loading bacpipe and its models, this can take a moment ...\n", flush=True)
    started = time.time()

    import bacpipe
    import bacpipe.core.workflows as workflows

    # same effect as editing config.yaml and settings.yaml
    for key, value in run["config"].items():
        setattr(bacpipe.config, key, value)
    for key, value in run["settings"].items():
        setattr(bacpipe.settings, key, value)

    def remember_dashboard_arguments(**kwargs):
        """Stand-in for ``visualize_using_dashboard``, see module docstring."""
        with open(run["dashboard_kwargs_file"], "wb") as f:
            pickle.dump(kwargs, f)

    workflows.visualize_using_dashboard = remember_dashboard_arguments

    bacpipe.play(bool_save_logs=True, **run["config"], **run["settings"])

    print(f"\nPipeline finished after {time.time() - started:.0f} s.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
