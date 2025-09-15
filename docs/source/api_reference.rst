API Reference
=============

This page lists all the objects that are part of the **public API** of
``bacpipe``. These are either defined in the package or explicitly
re-exported in ``__init__.py``.

.. toctree::
   :maxdepth: 3
   :hidden:

.. currentmodule:: bacpipe

Embedding
---------

.. autoclass:: Embedder
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Main Pipeline Functions
-----------------------

.. autofunction:: get_model_names
.. autofunction:: evaluation_with_settings_already_exists
.. autofunction:: model_specific_embedding_creation
.. autofunction:: model_specific_evaluation
.. autofunction:: cross_model_evaluation
.. autofunction:: visualize_using_dashboard

Core Functions
--------------

.. autofunction:: play
.. autofunction:: ensure_std_models
