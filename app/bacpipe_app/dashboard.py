"""
Build the bacpipe dashboard for a finished run and return its layout, so it
can be embedded in the application page instead of being served on its own
by ``visualize_using_dashboard``.
"""

import inspect
import pickle

#: names used from bacpipe by this module and worker.py, checked by tests/test_contract.py
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
    Create the dashboard for a finished run. Imports bacpipe.

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
