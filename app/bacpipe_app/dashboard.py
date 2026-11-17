"""
Stage 3: build the regular bacpipe dashboard so that it can be placed inside
the page of the application.

``bacpipe.visualize_using_dashboard`` builds the dashboard *and* starts its
own web server for it. The application already is a web server, so this
module does the first half only: it creates ``DashBoard`` with the arguments
the pipeline would have used (worker.py stored them) and returns its layout.

This is the only place where the application reaches into bacpipe internals.
The names it relies on are listed in ``BACPIPE_CONTRACT`` and checked by
``app/tests/test_contract.py`` on every push, so a rename in bacpipe is
noticed in the normal test run and not by a user of the application.
"""

import inspect
import pickle

#: what this module (and worker.py) needs from bacpipe: {file: [names]}
BACPIPE_CONTRACT = {
    "core/workflows.py": ["play", "visualize_using_dashboard"],
    "embedding_evaluation/visualization/dashboard.py": [
        "DashBoard",
        "DashBoard.build_layout",
        "visualize_using_dashboard",
    ],
    "core/constants.py": ["supported_models"],
}


def build_dashboard(config, settings, dashboard_kwargs_file):
    """
    Create the dashboard for a finished run.

    This imports bacpipe (torch, tensorflow, ...) into the application
    process, which takes a few seconds the first time.

    Parameters
    ----------
    config, settings : dict
        the values the pipeline ran with
    dashboard_kwargs_file : pathlib.Path
        file written by worker.py with the arguments of
        ``visualize_using_dashboard``

    Returns
    -------
    panel.Tabs
        the layout of the dashboard (``DashBoard.app``)
    """
    import bacpipe
    from bacpipe.embedding_evaluation.visualization import dashboard as module

    # the dashboard reads some values directly from these namespaces
    for key, value in config.items():
        setattr(bacpipe.config, key, value)
    for key, value in settings.items():
        setattr(bacpipe.settings, key, value)

    with open(dashboard_kwargs_file, "rb") as f:
        kwargs = pickle.load(f)

    # visualize_using_dashboard consumes its named parameters (the server
    # port, address, ...) and hands `models` plus the rest to DashBoard
    named = inspect.signature(module.visualize_using_dashboard).parameters
    models = kwargs.pop("models")
    for name, parameter in named.items():
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD:
            kwargs.pop(name, None)

    dashboard = module.DashBoard(models, **kwargs)
    dashboard.build_layout()
    return dashboard.app
