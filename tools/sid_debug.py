import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import rawpy
import torch
from PIL import Image
from models.sony_images import sid_original, dataset

in_image_file = './../dataset/Sony/short/10003_00_0.04s.ARW'
activations_file = './../layer_activations_10003_00_250.npz'

activations = np.load(activations_file)

model = sid_original.Model()
model.load_state()

raw = rawpy.imread(in_image_file)
ratio = 250
in_image = torch.unsqueeze(dataset.pack_raw(raw, ratio), 0)

with torch.no_grad():
    out = model(in_image)

for i in range(0, len(model.activations)):
    torch_tensor = model.activations[i].numpy()
    tf_tensor = activations[f'arr_{i}']
    tf_tensor = np.permute_dims(tf_tensor, axes=(0, 3, 1, 2)) # all in shape (N, C, H, W)
    print(f'{i} {torch_tensor.shape} {tf_tensor.shape}')
    
    dist = ((torch_tensor - tf_tensor)**2).max()
    print(dist)
    
    # diff maps
    if i == 0 and False:
        diff = tf_tensor - torch_tensor
        abs_diff = np.abs(diff)
        max_index = np.unravel_index(np.argmax(abs_diff), diff.shape)
        max_val = abs_diff[max_index]
        print(max_index)
        print(f'tf {tf_tensor[max_index]}')
        print(f'torch {torch_tensor[max_index]}')
        
        diff_maps = ((diff / max_val + 1) * 255.0 / 2.0).astype(np.uint8).squeeze(0)
        for j, map in enumerate(diff_maps):
            img = Image.fromarray(map)
            img.save(f'./diff_map/map_({max_val})_{j}.png')
        