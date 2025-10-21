import torch

# Load the original checkpoint
checkpoint = torch.load("sid_bottleneck_transformer_retrained_4b_c.pt", map_location='cpu')

# Get the state_dict
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Create a new state_dict with renamed keys
new_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("bottleneck5.conv"):
        new_key = key.replace("bottleneck5.conv", "bottleneck5.up.0")
    elif key.startswith("bottleneck5.up"):
        new_key = key.replace("bottleneck5.up", "bottleneck5.up.1")
    else:
        new_key = key
    new_state_dict[new_key] = value

# Optionally: Save the fixed state_dict
torch.save(new_state_dict, "sid_bottleneck_transformer_retrained_4b_c.pt")

# Or load it into your model
# model.load_state_dict(new_state_dict)
