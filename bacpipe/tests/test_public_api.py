"""
Unit tests for the top-level ``bacpipe`` public API surface.

These cover the API-inspection cells of the ``simple_use_cases.ipynb``
example notebook (``bacpipe.__all__``, ``bacpipe.supported_models``,
``bacpipe.EMBEDDING_DIMENSIONS`` and ``bacpipe.get_audio_files``).
"""

import bacpipe


EXPECTED_PIPELINES = {
    "play",
    "run_pipeline_for_single_model",
    "run_pipeline_for_models",
    "generate_embeddings",
}
EXPECTED_LABEL_HELPERS = {
    "metadata_labels",
    "ground_truth_by_model",
    "get_dt_filename",
}
EXPECTED_PROBING = {"probing_pipeline", "run_probe_inference", "prepare_probe_inference"}
EXPECTED_CLUSTERING = {
    "clustering_pipeline",
    "run_clustering",
    "eval_clustering",
    "eval_with_silhouette",
}


class TestPublicApi:
    def test_all_pipelines_are_exported(self):
        assert EXPECTED_PIPELINES <= set(bacpipe.__all__)

    def test_label_helpers_are_exported(self):
        assert EXPECTED_LABEL_HELPERS <= set(bacpipe.__all__)

    def test_probing_helpers_are_exported(self):
        assert EXPECTED_PROBING <= set(bacpipe.__all__)

    def test_clustering_helpers_are_exported(self):
        assert EXPECTED_CLUSTERING <= set(bacpipe.__all__)

    def test_loader_and_embedder_are_exported(self):
        assert {"Loader", "Embedder"} <= set(bacpipe.__all__)

    def test_all_exported_names_are_attributes(self):
        for name in bacpipe.__all__:
            assert hasattr(bacpipe, name)


class TestSupportedModels:
    def test_is_non_empty_list(self):
        assert isinstance(bacpipe.supported_models, list)
        assert len(bacpipe.supported_models) > 0

    def test_contains_birdnet(self):
        assert "birdnet" in bacpipe.supported_models


class TestEmbeddingDimensions:
    def test_is_dict_keyed_by_supported_models(self):
        assert isinstance(bacpipe.EMBEDDING_DIMENSIONS, dict)
        for model in bacpipe.supported_models:
            assert model in bacpipe.EMBEDDING_DIMENSIONS

    def test_dimensions_are_positive_ints(self):
        for dim in bacpipe.EMBEDDING_DIMENSIONS.values():
            assert isinstance(dim, int)
            assert dim > 0


class TestGetAudioFilesAlias:
    def test_top_level_alias_matches_loader_method(self):
        assert bacpipe.get_audio_files == bacpipe.Loader.get_audio_files

    def test_finds_audio_files_in_test_data(self):
        files = bacpipe.get_audio_files(
            "bacpipe/tests/test_data/audio", return_type="str"
        )
        assert len(files) > 0
        assert all(f.endswith((".wav", ".WAV", ".flac", ".mp3")) for f in files)
