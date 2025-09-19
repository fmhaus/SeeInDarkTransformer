import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from models.sony_images import sid_original, sid_bottleneck_transformer, sid_no_bottleneck
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torchprofile
import gc
import time
from tqdm import tqdm

def get_random_input(device):
    return torch.randn(1, 4, 2848 // 2, 4256 // 2, device=device)

def print_profile(model, name, n_runs = 10):
    use_cuda = torch.cuda.is_available()
    
    print(sum(param.numel() if '5' in name else 0 for name, param in model.named_parameters()))
    
    for n, param in model.named_parameters():
        if '5' in n:
            print(f"{n} {param.numel()}")
    
    params = sum(param.numel() for param in model.parameters())
    model.eval()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    print(f'Using device {device}.')
    
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    with torch.no_grad():
        input = get_random_input(device)
        macs = torchprofile.profile_macs(model, input)
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True) as prof:
            
            with record_function("model_inference"):
                
                output = model(input)
                
                if use_cuda:
                    torch.cuda.synchronize()
    
    
        time_begin = time.time()
        for i in tqdm(range(n_runs), f'Testing {name}'):
            out = model(get_random_input(device))
        
        if use_cuda:
            torch.cuda.synchronize()
            
        time_passed = (time.time() - time_begin) / n_runs

    print(f"Model '{name}'")
    print(f'Parameters: {params}')
    print(f'MACs: {macs}')
    print(f'Average time: {time_passed}s')
    
    #print(prof.key_averages().table(row_limit=100))

if __name__ == "__main__":
    print_profile(sid_original.Model(), 'sid_original', 1)
    #print_profile(sid_no_bottleneck.Model(), 'sid_no_bottleneck')
    print_profile(sid_bottleneck_transformer.Model(), 'sid_bottleneck_transformer', 1)