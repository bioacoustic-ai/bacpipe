import bacpipe
from bacpipe import (
    make_set_paths_func,
    ground_truth_by_model,
    probing_pipeline,
    clustering_pipeline,
    EMBEDDING_DIMENSIONS,
    run_pipeline_for_single_model,
)

embeddings = {}

# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------
# Remove all the module-level config loading and replace with:
embeddings = {}

import librosa as lb
from bacpipe.model_pipelines.model_utils import ModelBaseClass

class MyModel(ModelBaseClass):
    SAMPLE_RATE = 48000         # the sample rate of the audio files that will be processed by the model
    SEGMENT_LENGTH = 48000*3    # 3s => the length of the audio segments that will be processed by the model (in samples)

    def __init__(self, **kwargs):
        super().__init__(sr=self.SAMPLE_RATE, segment_length=self.SEGMENT_LENGTH, **kwargs)

    def preprocess(self, audio):
        return audio

    def __call__(self, audio):
        audio = audio.cpu().numpy()
        mel_spec = lb.feature.melspectrogram(y=audio, sr=self.SAMPLE_RATE)
        # return array needs to be 2D!
        mel_spec = mel_spec.reshape(
            [len(mel_spec), mel_spec.shape[-2] * mel_spec.shape[-1]]
            )
        return mel_spec

def test_embedding_generation(
    kwargs,
):
    # bacpipe.ensure_models_exist(
    #     bacpipe.settings.model_base_path, model_names=[model]
    # )
    # model = bacpipe.confirm_model_name(model)

    embeddings = run_pipeline_for_single_model(
        model_name='insect66', **kwargs
    )

    assert embeddings.files, f"No embeddings generated for "


# def test_embedding_dimensions(model, kwargs):
#     model = bacpipe.confirm_model_name(model)
#     assert (
#         embeddings[model].metadata_dict["embedding_size"]
#         == EMBEDDING_DIMENSIONS[model]
#     ), f"Embedding dimension mismatch for {model}"


# def test_evaluation(model, overwrite, device, only_embed_annotations, kwargs):
#     model = bacpipe.confirm_model_name(model)
#     embeds = embeddings[model].embeddings(return_type="array")
#     get_paths = make_set_paths_func(**kwargs)
#     paths = get_paths(model)
#     if model in bacpipe.TF_MODELS:
#         kwargs["device"] = "cpu"
#     try:
#         ground_truth = ground_truth_by_model(
#             model, **kwargs
#         )
#     except FileNotFoundError:
#         ground_truth = None
#     assert len(embeds) > 1

#     if overwrite:
#         if (paths.labels_path / "probing_dataframe.csv").exists():
#             (paths.labels_path / "probing_dataframe.csv").unlink()
#     for class_config in kwargs.get("probe_configs", {}).values():
#         if class_config["bool"]:
#             probing_pipeline(
#                 model,
#                 ground_truth,
#                 embeds,
#                 paths,
#                 **class_config,
#                 **kwargs,
#             )
#     clustering_pipeline(model, ground_truth, embeds, paths, **kwargs)


# def test_collecting_embeddings(model):
#     model = bacpipe.confirm_model_name(model)
#     embeds = embeddings[model].embeddings(return_type="array")
#     assert len(embeds) > 0
#     embeds = embeddings[model].embeddings(return_type="dict")
#     assert len(embeds) > 0


# def test_collecting_predictions(model):
#     model = bacpipe.confirm_model_name(model)
#     try:
#         ar_preds = embeddings[model].predictions(return_type="array")
#         df_preds = embeddings[model].predictions(return_type="dataframe")
#         di_preds = embeddings[model].predictions(return_type="dict")
#         assert len(ar_preds) > 0
#         assert len(df_preds) > 0
#         assert len(di_preds) > 0
#     except FileNotFoundError:
#         pass


# def test_benchmarking(model, device, only_embed_annotations, kwargs):
#     model = bacpipe.confirm_model_name(model)
#     try:
#         results = bacpipe.benchmark(
#             model,
#             kwargs["audio_dir"],
#             check_if_already_processed=True,
#             annotations_file="annotations.csv",
#         )
#         assert isinstance(results, dict)
#     except AttributeError as e:
#         print(e)
