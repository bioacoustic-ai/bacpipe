import bacpipe

if __name__ == "__main__":
    # Run with defaults
    
    bacpipe.config.models=['birdmae', 'naturebeats']
    bacpipe.config.already_computed=True
    # bacpipe.settings.device='cuda'
    bacpipe.settings.model_base_path = '/home/siriussound/Code/bacpipe/bacpipe/model_checkpoints'
    bacpipe.config.evaluation_task=['classification', 'clustering']
    bacpipe.settings.testing = True
    bacpipe.config.overwrite = True
    bacpipe.settings.main_results_dir='bacpipe/tests/results_files'
    
    bacpipe.play(save_logs=True)

    # To modify config or settings you can use the following:
    # bacpipe.config.audio_dir = "data/audio"
    # bacpipe.settings.main_results_dir = "results/"
    # bacpipe.play(save_logs=True)
    
    # But it's probably easier if you just modify the config.yaml or bacpipe/settings.yaml files
