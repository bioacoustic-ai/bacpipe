from pathlib import Path
import json

embeds_ar = []
files = Path('bacpipe/test_files/umap_embeds/2024-10-25_15-20___umap-testing_embeds-passt').iterdir()
for file in files:
    if file.suffix == '.json':
        with open(file, 'r') as f:
            embeds_ar.append(json.load(f))
        
        
import matplotlib.pyplot as plt

x0 = embeds_ar[0]['x']
y0 = embeds_ar[0]['y']
x1 = embeds_ar[1]['x']
y1 = embeds_ar[1]['y']

plt.figure()
plt.plot(x0, y0, 'o', label='humpback')
plt.plot(x1, y1, 'x', label='bird')
plt.legend()
plt.savefig('test.png')