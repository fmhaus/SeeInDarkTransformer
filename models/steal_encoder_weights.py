import numpy as np
import torch
import os
from sony_images.sid_bottleneck_transformer import Model

EXPORT_DIR = './models/Sony/tf_export/'
MODEL_STATE_FILE= './models/Sony/bottleneck_transformer/states/initial.pt'

NAME_MAP = {
    # Convolutional layers
    'g_conv1_1\\weights.npy': 'conv1_1.weight',
    'g_conv1_1\\biases.npy': 'conv1_1.bias',
    'g_conv1_2\\weights.npy': 'conv1_2.weight',
    'g_conv1_2\\biases.npy': 'conv1_2.bias',
    
    'g_conv2_1\\weights.npy': 'conv2_1.weight',
    'g_conv2_1\\biases.npy': 'conv2_1.bias',
    'g_conv2_2\\weights.npy': 'conv2_2.weight',
    'g_conv2_2\\biases.npy': 'conv2_2.bias',
    
    'g_conv3_1\\weights.npy': 'conv3_1.weight',
    'g_conv3_1\\biases.npy': 'conv3_1.bias',
    'g_conv3_2\\weights.npy': 'conv3_2.weight',
    'g_conv3_2\\biases.npy': 'conv3_2.bias',
    
    'g_conv4_1\\weights.npy': 'conv4_1.weight',
    'g_conv4_1\\biases.npy': 'conv4_1.bias',
    'g_conv4_2\\weights.npy': 'conv4_2.weight',
    'g_conv4_2\\biases.npy': 'conv4_2.bias'
}

model = Model()
state_dict = model.state_dict()

for root, dirs, files in os.walk(EXPORT_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        name = file_path[len(EXPORT_DIR):]
        tensor = np.load(file_path, allow_pickle=True)
    
        if name in NAME_MAP:
            model_name = NAME_MAP[name]
            if 'weight' in model_name:
                state_dict[model_name] = torch.from_numpy(tensor).permute(3, 2, 0, 1)
            else:
                state_dict[model_name] = torch.from_numpy(tensor)

model.load_state_dict(state_dict)

torch.save(state_dict, MODEL_STATE_FILE)
print(f'Saved to {MODEL_STATE_FILE}')