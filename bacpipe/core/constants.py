

TF_MODELS = [
    'birdnet', 
    'perch_v2',
    'perch_bird', 
    'google_whale', 
    'surfperch', 
    'vggish',
    'hbdet', 
]
"""list[str]: models that were built using tensorflow"""

EMBEDDING_DIMENSIONS = {
    "audiomae": 768,
    "audioprotopnet": 1024,
    "avesecho_passt": 768,
    "aves_especies": 768,
    "bat": 64,
    "batdetect2_clip_avg": 32,
    "batdetect2_dets_avg": 32,
    "beats": 768,
    "birdaves_especies": 1024,
    "biolingual": 512,
    "birdnet": 1024,
    "birdmae": 1280,
    "convnext_birdset": 1024,
    "hbdet": 2048,
    "insect66": 1280,
    "insect459": 1280,
    "mix2": 960,
    "naturebeats": 768,
    "perch_bird": 1280,
    "perch_v2": 1536,
    "protoclr": 384,
    "rcl_fs_bsed": 2048,
    "surfperch": 1280,
    "google_whale": 1280,
    "vggish": 128,
}
"""dict: model names with their corresponding embedding size"""

NEEDS_CHECKPOINT = [
    "audiomae",
    "avesecho_passt",
    "aves_especies",
    "bat",
    "beats",
    "birdaves_especies",
    "birdnet",
    "hbdet",
    "insect66",
    "insect459",
    "mix2",
    "naturebeats",
    "protoclr",
    "rcl_fs_bsed"
]
"""list[str]: models that require a checkpoint to be downloaded to be run"""


supported_models = list(EMBEDDING_DIMENSIONS.keys())
"""list[str]: Supported embedding models available in bacpipe."""

models_needing_checkpoint = NEEDS_CHECKPOINT
"""list[str]: Models that require a checkpoint to be downloaded before use."""

