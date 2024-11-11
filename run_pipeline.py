from bacpipe.umap_embed import get_embeddings
import yaml

with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)

for model_name in config["embedding_model"]:
    embeddings, metadata_dict, divisions_array = get_embeddings(model_name = model_name, 
                                                                audio_dir = config["audio_dir"])
