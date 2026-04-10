API Reference
=============

This page lists all the objects that are part of the **public API** of
``bacpipe``. These are either defined in the package or explicitly
re-exported in ``__init__.py``.

.. toctree::
   :maxdepth: 3
   :hidden:

.. currentmodule:: bacpipe

Constants
--------------

.. autofunction:: supported_models 
.. autofunction:: models_needing_checkpoint
.. autofunction:: TF_MODELS
.. autofunction:: EMBEDDING_DIMENSIONS
.. autofunction:: NEEDS_CHECKPOINT

Main Processing Classes
---------

.. autoclass:: Loader
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: Embedder
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Main Pipeline Functions
-----------------------

.. autofunction:: play
.. autofunction:: run_pipeline_for_single_model
.. autofunction:: run_pipeline_for_models
.. autofunction:: generate_embeddings
    
Return audio files in specified dir
--------------

.. autofunction:: get_audio_files

Automatic creation of labels and ground truth
--------------

.. autofunction:: DefaultLabels 
.. autofunction:: create_default_labels
.. autofunction:: ground_truth_by_model
.. autofunction:: get_default_labels
.. autofunction:: get_dt_filename

Probing functions
--------------

.. autofunction:: probing_pipeline
.. autofunction:: run_probe_inference
.. autofunction:: prepare_probe_inference

Clustering functions
--------------

.. autofunction:: clustering_pipeline
.. autofunction:: run_clustering
.. autofunction:: eval_clustering
.. autofunction:: eval_with_silhouette

Evaluation pipelines
--------------

.. autofunction:: benchmark
.. autofunction:: model_specific_evaluation
.. autofunction:: cross_model_evaluation

Experiment managing functions
--------------

.. autofunction:: ensure_models_exist
.. autofunction:: evaluation_with_settings_already_exists
.. autofunction:: get_model_names
.. autofunction:: make_set_paths_func

Visualization function to start dashboard
--------------

.. autofunction:: visualize_using_dashboard

