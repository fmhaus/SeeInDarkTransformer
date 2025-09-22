import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from models.sony_images import sid_bottleneck_transformer_3b, sid_original, sid_no_bottleneck
import torch
import torchprofile
import os
import psutil
import gc
import time
from tqdm import tqdm
import numpy as np
import argparse

def get_random_input(device):
    return torch.randn(1, 4, 2848 // 2, 4256 // 2, device=device)

def get_model_params(model):
    return sum(param.numel() for param in model.parameters())

def get_model_macs(model, device):
    with torch.no_grad():
        input = get_random_input(device)
        return torchprofile.profile_macs(model, input)

def profile_forward_time_memory(model, device, n_runs):
    use_cuda = device.type == 'cuda'
    
    process = psutil.Process(os.getpid())
    gc.disable()
    gc.collect()
    
    times = np.empty((n_runs), dtype=np.float64)
    memory_main = np.empty((n_runs), dtype=np.uint64)
    memory_gpu = np.zeros((n_runs), dtype=np.uint64)
    
    with torch.no_grad():
        for i in tqdm(range(n_runs)):
            
            time_begin = time.perf_counter()
            
            input = get_random_input(device)
            _ = model(input)
        
            if use_cuda:
                torch.cuda.synchronize(device)
            
            times[i] = time.perf_counter() - time_begin
            memory_main[i] = process.memory_info().rss / (1024**2)
            if use_cuda:
                memory_gpu[i] = torch.cuda.max_memory_allocated(device) / (1024**2)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(device)
            
            gc.collect()
        
    gc.enable()
    
    return times, memory_main, memory_gpu

class Benchmark:
    def __init__(self, model, device, n_runs = 10, compile_model = False):
        self.name = f'{model.__class__.__module__}.{model.__class__.__name__}'
        self.params = get_model_params(model)
        
        model.to(device)
    
        self.macs = get_model_macs(model, device)
        
        if compile_model:
            model = torch.compile(model)
            with torch.no_grad():
                model(get_random_input(device))
        
        self.times, self.main_ram, self.vram = profile_forward_time_memory(model, device, n_runs)

    def print_results(self):
        print("--------------------------------")
        print(f"Model: {self.name}")
        print(f"Parameters: {self.params}")
        print(f"MACs: {self.macs}")
        print(f"Time (avg, min, max) (s): {self.times.mean()}, {self.times.min()}, {self.times.max()}")
        print(f"System memory (avg, min, max) (MB): {self.main_ram.mean()}, {self.main_ram.min()}, {self.main_ram.max()}")
        print(f"GPU memory (avg, min, max) (MB): {self.vram.mean()}, {self.vram.min()}, {self.vram.max()}")

def init_options(parser):
    parser.add_argument('--compile', action='store_true', default=False, help='Compiles the model before benchmarking')
    parser.add_argument('--n_runs', type=int, default=10, help='How many inference runs for benchmark')
    return parser

if __name__ == "__main__":
    opt = init_options(argparse.ArgumentParser()).parse_args()
    if opt.compile:
        compile_model = True
    else:
        compile_model = False
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    original = Benchmark(sid_original.Model(), device, opt.n_runs, compile_model)
    transformer = Benchmark(sid_bottleneck_transformer_3b.Model(), device, opt.n_runs,compile_model)
    
    print(f'Compile: {compile_model}')
    original.print_results()
    transformer.print_results()