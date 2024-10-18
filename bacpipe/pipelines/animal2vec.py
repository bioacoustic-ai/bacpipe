from ievad.animal2vec_nn.nn import chunk_and_normalize
from fairseq import checkpoint_utils
import torch

def get_animal2vec_preprocessing(self, audio):
    chunk = chunk_and_normalize(
        torch.tensor(audio),
        segment_length=10,
        sample_rate=8000,
        normalize=True,
        max_batch_size=16
    )
    # if not torch.is_tensor(chunk):
    #     chunk = torch.stack(chunk)  # stack the list of tensors
    # elif chunk.dim() == 1:  # split segments or single segment
    #     chunk = chunk.view(1, -1)
    return chunk
    
def get_callable_animal2vec_model(self, **kwargs):
    path_to_pt_file = "ievad/models/animal2vec/animal2vec_large_finetuned_MeerKAT_240507.pt"
    models, _ = checkpoint_utils.load_model_ensemble([path_to_pt_file])
    model = models[0].to("cpu")
    model.eval()
    
    @torch.inference_mode()
    def a2v_infer(samples):
        all_embeds = []
        for batch in tqdm(samples[0]):
            res = model(source=batch.view(1, -1))
            embeds = res['layer_results']
            np_embeds = [a.detach().numpy() for a in embeds]
            all_embeds.append(np_embeds)
        return np.array(all_embeds)
    
    return a2v_infer