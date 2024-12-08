from bacpipe.main import get_embeddings
from bacpipe.visualize import plot_comparison
import yaml

with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)

for model_name in config["embedding_model"]:
    get_embeddings(
        model_name=model_name,
        dim_reduction_model=config["dim_reduction_model"],
        audio_dir=config["audio_dir"],
    )
plot_comparison(
    config["audio_dir"], config["embedding_model"], config["dim_reduction_model"]
)
