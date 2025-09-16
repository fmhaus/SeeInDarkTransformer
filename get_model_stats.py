from models.sony_images import sid_original, sid_bottleneck_transformer
import torch
import torchprofile
import time
from tqdm import tqdm

def print_profile(model, name, n_runs = 10):
    params = sum(param.numel() for param in model.parameters())
    model.eval()

    with torch.no_grad():
        input = torch.randn(1, 4, 2848 // 2, 4256 // 2)
        macs = torchprofile.profile_macs(model, input)
        
        time_begin = time.time()
        for i in tqdm(range(n_runs), f'Testing {name}'):
            input = torch.randn(1, 4, 2848 // 2, 4256 // 2)
            out = model(input)
        time_passed = (time.time() - time_begin) / n_runs

    print(f"Model '{name}'")
    print(f'Parameters: {params}')
    print(f'MACs: {macs}')
    print(f'Average time: {time_passed}s')

if __name__ == "__main__":
    print_profile(sid_original.Model(), 'sid_original')
    print_profile(sid_bottleneck_transformer.Model(), 'sid_bottleneck_transformer')