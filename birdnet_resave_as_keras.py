import tensorflow as tf
import keras


class Rebuilder:
    def __init__(self, model):
        self.input_layer = tf.keras.Input(shape=model.layers[4].input.shape[1:], name='inputs', dtype=tf.float32)
        # we're starting at layer 4 because the MelSpecLayerSimple classes wont be deserializable by keras 3.11
        # but we don't need them because they are just part of the preprocessing anyway
        # So essentially we're starting after the concatenation of the two spectrograms
        self.layer_map = {layer.name: layer for layer in model.layers[4:]} 
        self.output_cache = {}  # cache outputs to stop infinite recursion
        self.layer_confs = []

    def rebuild_layer(self, layer):
        if layer.name in self.output_cache:
            return self.output_cache[layer.name]
        
        layer_config = {
            "name": layer.name,
            "class_name": layer.__class__.__name__,
            "config": layer.get_config(),
            "inbound_nodes": []
        }
        
        if 'axis' in layer_config['config']:
            if not isinstance(layer_config['config']['axis'], int):
                layer_config['config']['axis'] = layer_config['config']['axis'][0]

        # Handle multiple inputs
        inbound = []
        if isinstance(layer.input, list):
            inputs = []
            for inp in layer.input:
                inbound.append(inp._keras_history.layer.name)
                inputs.append(self.rebuild_layer(inp._keras_history.layer))
        else:
            inp = layer.input
            inbound.append(inp._keras_history.layer.name)
            if inp.name.startswith("concat") or inp.name.startswith("INPUT"):
                layer_config['inbound_nodes'] = inbound
                
                out = layer(self.input_layer)
                self.output_cache[layer.name] = out
                self.layer_confs.append(layer_config)
                return out
            inputs = [self.rebuild_layer(inp._keras_history.layer)]

        layer_config['inbound_nodes'] = inbound
        
        out = layer(inputs if len(inputs) > 1 else inputs[0])
        self.output_cache[layer.name] = out
        self.layer_confs.append(layer_config)
        return out

    def build_model(self, model):
        # Recurse from final output layer
        output = self.rebuild_layer(model.layers[-1])
        return (
            {
                "input_shape": self.input_layer.shape,
                "layer_confs": self.layer_confs
                },
            keras.Model(self.input_layer, output, name="rebuilt_model")
        )


def resave_Birdnet_as_keras():
    import numpy as np
    if keras.__version__[0] == '2':
        # This is for the old environment to save the birdnet model in the 
        # tensorflow==2.15 and keras 2.15 version
        model = tf.keras.models.load_model(
            "bacpipe/model_checkpoints/birdnet", compile=False
            )
    
        a = Rebuilder(model.model)    
        dic, spec_input_model = a.build_model(model.model)
            
        ### save graph
        tf.saved_model.save(spec_input_model, 'concat_input_graph')
        ### save model_dict
        np.save('model_graph_dict.npy', dic, allow_pickle=True)
        
        ### creating a Preprocessor model to save as tflite file
        input_layer = keras.Input(shape=(144000,), name='input', dtype=tf.float32)
        preprocessor = keras.Model(input_layer, 
                                   model.model.layers[3]([
                                       model.model.layers[1](input_layer), 
                                       model.model.layers[2](input_layer)
                                       ]))
        tf.saved_model.save(preprocessor, "BirdNET_Preprocessor")
            
        a = tf.convert_to_tensor(np.zeros([1, 144000]))
        
        print('results if we pass a zero array to the spectrogram input model:', 
              spec_input_model(preprocessor(a)))
        print('results if we pass a zero array to the original model:', 
              model.model(a))
        print('they all pretty much match. so now we save the graph model and the dict'
              ' that way we can load them again on the other side with the higher versions.')
        
    else:
        # Here we load the model assuming we're now in an environment with
        # tensorflow==2.20 and keras==3.11 or higher
        ### load model dict
        dic = np.load('model_graph_dict.npy', allow_pickle=True).item()
        
        loaded_preprocessor = tf.saved_model.load('BirdNET_Preprocessor')
        preprocessor = lambda x: loaded_preprocessor.signatures['serving_default'](x)['concatenate']
        
        ### rebuilding model      
        layer_map = {}
        input_layer = keras.Input(shape=dic["input_shape"][1:], name="inputs")
        
        layer_map['concatenate'] = input_layer

        for idx, layer_config in enumerate(dic['layer_confs']):
            cls = getattr(keras.layers, layer_config['class_name'])
            unsupported_keys = ["groups"]  # add more if needed
            for key in unsupported_keys:
                layer_config['config'].pop(key, None)
            layer = cls.from_config(layer_config['config'])
            
            if not layer_config['inbound_nodes'] or layer_config['inbound_nodes'][0].startswith('input'):
                out = input_layer
            else:
                inputs = [layer_map[name] for name in layer_config['inbound_nodes']]
                out = layer(inputs if len(inputs) > 1 else inputs[0])
            layer_map[layer_config['name']] = out
        
        rebuilt = keras.Model(input_layer, out)
        print('This is now the new rebuilt model architecture: ')
        rebuilt.summary()
        
        print('it should match the old one, minus the preprocessing.')

        ### fitting graph weights to rebuilt model
        graph_model = tf.saved_model.load('concat_input_graph')
        ## ensure inference works
        import numpy as np

        a = tf.convert_to_tensor(np.zeros([3, 144000]), dtype=tf.float32)
        
        name_to_var = {v.name: v for v in graph_model.variables}
        
        for layer in rebuilt.layers:
            weights = []
            for w in layer.weights:
                # if w.name in name_to_var:
                var_str = f'{layer.name}/{w.name}:0'
                
                if var_str in name_to_var:
                    
                    weights.append(name_to_var[var_str].numpy())
                elif 'kernel' in var_str:
                    var_str = var_str.replace('kernel', 'depthwise_kernel')
                    weights.append(name_to_var[var_str].numpy())
                    print('replaced kernel with depthwise_kernel')
                else:
                    print(var_str, 'weight name not found ---> This will be a problem!')
            if weights:
                layer.set_weights(weights)

        out1 = graph_model.signatures['serving_default'](preprocessor(a))
        out2 = rebuilt(preprocessor(a))
        test_results = np.allclose(out1['CLASS_DENSE_LAYER'].numpy(), out2.numpy(), atol=1e-5)
        print(f'{test_results=}, This shows us that the graph model and the newly '
              'created keras api model match. we now save the model as a .keras model'
              ' so it can be imported in the higher tf and keras versions without a problem'
              ' and without needing this rebuilding.'
              )
        rebuilt.save('birdnetv2.4_keras3.keras')

resave_Birdnet_as_keras()