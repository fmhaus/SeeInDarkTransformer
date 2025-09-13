import numpy as np
import torch
import os
from models.sony_images import sid_bottleneck_transformer

EXPORT_DIR = './models/sony_images/states/tf_export/'
MODEL_STATE_FILE= './models/sony_images/states/sid_bottleneck_transformer_initial.pt'

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
    'g_conv4_2\\biases.npy': 'conv4_2.bias',
    
    # Skip 
    #'g_conv5_1\\weights.npy': 'conv5_1.weight',
    #'g_conv5_1\\biases.npy': 'conv5_1.bias',
    #'g_conv5_2\\weights.npy': 'conv5_2.weight',
    #'g_conv5_2\\biases.npy': 'conv5_2.bias',
    
    # Upconvolution/transpose convolution layers
    'Variable.npy': 'up6.weight',
    'g_conv6_1\\weights.npy': 'conv6_1.weight',
    'g_conv6_1\\biases.npy': 'conv6_1.bias',
    'g_conv6_2\\weights.npy': 'conv6_2.weight',
    'g_conv6_2\\biases.npy': 'conv6_2.bias',
    
    'Variable_1.npy': 'up7.weight',
    'g_conv7_1\\weights.npy': 'conv7_1.weight',
    'g_conv7_1\\biases.npy': 'conv7_1.bias',
    'g_conv7_2\\weights.npy': 'conv7_2.weight',
    'g_conv7_2\\biases.npy': 'conv7_2.bias',
    
    'Variable_2.npy': 'up8.weight',
    'g_conv8_1\\weights.npy': 'conv8_1.weight',
    'g_conv8_1\\biases.npy': 'conv8_1.bias',
    'g_conv8_2\\weights.npy': 'conv8_2.weight',
    'g_conv8_2\\biases.npy': 'conv8_2.bias',
    
    'Variable_3.npy': 'up9.weight',
    'g_conv9_1\\weights.npy': 'conv9_1.weight',
    'g_conv9_1\\biases.npy': 'conv9_1.bias',
    'g_conv9_2\\weights.npy': 'conv9_2.weight',
    'g_conv9_2\\biases.npy': 'conv9_2.bias',
    
    # Final output layer
    'g_conv10\\weights.npy': 'conv10.weight',
    'g_conv10\\biases.npy': 'conv10.bias'
}

model = sid_bottleneck_transformer.Model()
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