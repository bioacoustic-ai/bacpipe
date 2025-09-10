import bacpipe

if __name__ == "__main__":
    # Run with defaults

    # To modify config or settings you can use the following:
    # bacpipe.config.audio_dir = "data/audio"
    # this will specify that the audio data is in 'data/audio'
    
    # bacpipe.settings.main_results_dir = "../bacpipe_results"
    # this will ensure results are saved in the directory '../bacpipe_results'

    # bacpipe.config.models = ['birdmae', 'naturebeats']
    # this will run the models birdmae and naturebeats, for which you will have 
    # to download the checkpoints first (see ReadMe file)
        
    # bacpipe.play(save_logs=True)
    # this will save log files, configs and settings, which can be helpful
    # to retrace your steps if something malfunctions
    
    # But it's probably easier if you just modify the config.yaml or bacpipe/settings.yaml files
    
    
    bacpipe.play()
    
