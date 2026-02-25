import bacpipe


def print_output(var):
    print(f'\n{var=}\n')

if False:

    ### load configurations
    print(bacpipe.config)
    
    ### load more advanced settings
    print(bacpipe.settings)

    ### show all bacpipe api endpoints
    print_output(bacpipe.__all__)

    # get supported models
    print_output(bacpipe.supported_models)
    # get tensorflow models 
    print_output(bacpipe.TF_MODELS)
    # get embedding dimensions
    print_output(bacpipe.EMBEDDING_DIMENSIONS)

    ####################################################
    # get audio files in directory
    # loader_obj = bacpipe.Loader('bacpipe/tests/test_data/audio')
    # DONE
    audio_files = bacpipe.get_audio_files(
        'bacpipe/tests/test_data', return_as='str'
        )
    print_output(audio_files)

    #######################################################
    # get dt from audio files
    # DONE
    dt_files = []
    for file in audio_files:
        dt_files.append(bacpipe.get_dt_filename(file))
        
    print_output(dt_files)


    #######################################################
    # get multilabel gruond truth fitted to timestamps


    gt = bacpipe.ground_truth_by_model(
        'birdnet', 
        audio_dir='bacpipe/tests/test_data',
        annotations_filename='annotations.csv',
        single_label=False
        )

    print_output(gt)

    #######################################################
    # get default labels

    dl = bacpipe.create_default_labels(
        'bacpipe/tests/test_data',
        'birdnet'
        )

    print_output(dl)

    #######################################################
    # load single embeddings

    # create a loader object
    # you can create it from just the audio_dir, however, then
    # no folders will be created and embeddings can just be directly returned
    # not saved. We advise using bacpipe's workflows for improved performance.
    loader_obj = bacpipe.Loader('bacpipe/tests/test_data')

    # create a model embedder object
    embed_obj = bacpipe.Embedder('birdnet', loader_obj)

    embeds = embed_obj.get_embeddings_from_model(loader_obj.files[0])
    print_output(embeds)
    # will be empty because this only loads embeddings
    print_output(loader_obj.embeddings()) 
    print_output(embed_obj.classifier.predictions)

    #######################################################################
    # process all audio files in directory
    
    # create a new loader and model embedder object
    loader_obj = bacpipe.Loader('bacpipe/tests/test_data', 'birdnet', build_results_dir=True)
    embed_obj = bacpipe.Embedder('birdnet', loader_obj)
    # this will now generate embeddings and classification predictions for all the 
    # audio files in the specified directory
    embed_obj.run_inference_pipeline_using_multithreading()

    print_output(loader_obj.metadata_dict)
    print_output(loader_obj.embeddings())
    print_output(embed_obj.classifier.cumulative_annotations)

    #######################################################################
    # use one of bacpipe's workflows:

    loader_obj = bacpipe.generate_embeddings(
        model_name='birdnet',
        audio_dir='bacpipe/tests/test_data'
    )
    # like before
    print_output(loader_obj.metadata_dict)
    print_output(loader_obj.embeddings())

    ######################################################################
    # add your own model

    loader_obj = bacpipe.Loader('bacpipe/tests/test_data', 'mymodel')
    print_output(loader_obj.files)

    from bacpipe.model_pipelines.model_utils import ModelBaseClass


    import librosa as lb
    class MyModel(ModelBaseClass):
        SAMPLE_RATE = 48000
        SEGMENT_LENGTH = 48000
        def __init__(self, **kwargs):
            super().__init__(sr=self.SAMPLE_RATE, segment_length=self.SEGMENT_LENGTH, **kwargs)
            
        def preprocess(self, audio):
            return audio * 10
        
        def __call__(self, audio):
            audio = audio.cpu().numpy()
            mel_spec = lb.feature.melspectrogram(y=audio, sr=self.SAMPLE_RATE)
            return mel_spec
        
    embed_obj = bacpipe.Embedder('mymodel', loader_obj, device='cuda', Model=MyModel)
    mel_spectrograms = embed_obj.get_embeddings_from_model(loader_obj.files[0])

    #####################################################################
    # of even easier 

    loader_obj = bacpipe.generate_embeddings(
        model_name='mymodel',
        audio_dir='bacpipe/tests/test_data',
        Model=MyModel
    )

    print_output(loader_obj.metadata_dict)
    print_output(loader_obj.embeddings())

    # or load an existing model and modify it
    from bacpipe.model_pipelines.feature_extractors.birdnet_2 import Model

    class MyBirdNETModel(Model):
        def __call__(self, input):
            input = input**2
            return self.embeds(input, training=False)

    # loader_obj = bacpipe.generate_embeddings(
    #     model_name='birdnet2',
    #     audio_dir='bacpipe/tests/test_data',
    #     Model=MyBirdNETModel
    # )

    #########################################
    # even higher level workflow will run 
    # both the embedding and classification generation
    # and also dimensional reduction and plot the results

    loader_obj = bacpipe.run_pipeline_for_single_model(
        model_name='birdnet',
        audio_dir='bacpipe/tests/test_data',
        dim_reduction_model='umap'
    )



    # or with your own model:
    loader_obj = bacpipe.run_pipeline_for_single_model(
        model_name='birdnet2', # give your model a name
        audio_dir='bacpipe/tests/test_data',
        Model=MyBirdNETModel
    )



    ###########################################
    # even higher level is running a bunch of models
    # with dimensionality reduction on all of the audio
    # files in a given directory:

    loader_dictionary = bacpipe.run_pipeline_for_models(
        models=['birdnet', 'naturebeats'],
        audio_dir='bacpipe/tests/test_data',
        dim_reduction_model='umap'
        )

    print_output(loader_dictionary['birdnet'].metadata_dict)
    print_output(loader_dictionary['naturebeats'].embeddings())


    ############################################
    # now the same with custom models

    from bacpipe.model_pipelines.model_utils import ModelBaseClass
    import librosa as lb
    class MyModel(ModelBaseClass):
        SAMPLE_RATE = 48000
        SEGMENT_LENGTH = 48000
        def __init__(self, **kwargs):
            super().__init__(sr=self.SAMPLE_RATE, segment_length=self.SEGMENT_LENGTH, **kwargs)
            
        def preprocess(self, audio):
            return audio * 10
        
        def __call__(self, audio):
            audio = audio.cpu().numpy()
            mel_spec = lb.feature.melspectrogram(y=audio, sr=self.SAMPLE_RATE)
            mel_spec = mel_spec[:, :, 0]
            return mel_spec

    from bacpipe.model_pipelines.feature_extractors.birdnet_2 import Model

    class MyBirdNETModel(Model):
        def __call__(self, input):
            input = input**2
            return self.embeds(input, training=False)

    loader_dictionary = bacpipe.run_pipeline_for_models(
        models=['birdnet', 'mymodel', 'birdnet2'],
        audio_dir='bacpipe/tests/test_data',
        dim_reduction_model='umap',
        CustomModels=[None, MyModel, MyBirdNETModel]
        )

    #######################################
    # run whole pipeline 
    # and finally the whole bacpipe pipeline including
    # evaluation and dashboard visualization in the end

    bacpipe.play(
        models=['birdnet', 'mymodel', 'birdnet2'],
        audio_dir='bacpipe/tests/test_data',
        dim_reduction_model='umap',
        CustomModels=[None, MyModel, MyBirdNETModel]
    )




# get probing from model

print('hi')
# get clustering from model 

# ensure model exists, if not download



# benchmark HSN dataset using birdnet

