import numpy as np
from pathlib import Path

e_dir = 'frontend/public/files/embeds'
u_dir = 'frontend/public/files/umap_embeds'

# aves_embeds = [f for f in Path(e_dir).iterdir() if 'aves' in f.stem]
# aves_umap_embeds = [f for f in Path(u_dir).iterdir() if 'vggish' in f.stem]
birdnet_umap_embeds = [f for f in Path(u_dir).iterdir() if 'birdnet' in f.stem]

if False:
    embed_arrs = {}
    for d in aves_embeds:
        for f in [i for i in d.iterdir() if i.suffix == '.npy']:
            em = np.load(f)
            embed_arrs.update({f: em})

umap_embed_arrs = {}
for u in birdnet_umap_embeds:
    for f in [i for i in u.iterdir() if i.suffix == '.npy']:
        em = np.load(f)
        umap_embed_arrs.update({f: em})
    
print('fin')

import bacpipe.pipelines.umap as umap
umap.UMAP()