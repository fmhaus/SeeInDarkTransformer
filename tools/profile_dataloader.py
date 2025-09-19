import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.sony_images import dataset
from util import profiler
from multiprocessing import Manager
from util import image_util

import argparse
import options

if __name__ == '__main__':
    
    opt = options.Options().init(argparse.ArgumentParser()).parse_args()

    dataset.preprocess_raw_gts(os.path.join(opt.dataset_folder, 'Sony', 'long'), opt.preprocess_folder)
    
    with open('./data_lists/Sony_val_list.txt') as fr:
        val_list = list(line.split(' ') for line in fr.readlines())
        
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {'cuda' if torch.cuda.is_available() else 'cpu'}.')
    
    manager = Manager()
    profiling = profiler.ProcessProfiling(manager, ['read_raw', 'pack_raw', 'load_gt', 'augment'])
    main_profiler = profiler.Profiler()
    
    dataset_val = dataset.RawImageDataset(val_list, opt.dataset_folder, opt.preprocess_folder, pack_augment_on_worker=True, profiling = profiling)
   
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=opt.validation_batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        pin_memory=use_cuda, 
        drop_last=False, 
        persistent_workers=False
    )

    dataset_val.transform = image_util.augment_translate_new
  
    for ((raw, pack_settings), gt_images) in tqdm(dataloader_val, 'Pack on worker (augment_new)'):
        pass
    
    profiling.print_evaluation()
    profiling.reset_timings()
    
    dataset_val.transform = image_util.augment_translate_old
    
    for ((raw, pack_settings), gt_images) in tqdm(dataloader_val, 'Pack on worker (augment_old)'):
        pass
    
    profiling.print_evaluation()
    profiling.reset_timings()
    
    dataset_val.pack_augment_on_worker = False
    
    for ((raw_images, pack_settings), gt_images) in tqdm(dataloader_val, 'Pack on device'):
        
        with torch.no_grad():
            raw_images = raw_images.to(device, non_blocking=True)
            pack_settings = {key: value.to(device, non_blocking=True) for key, value in pack_settings.items()}
            gt_images = gt_images.to(device, non_blocking=True)
            
            main_profiler.section('pack_raw_main')
            packed_images = dataset.pack_raw(raw_images, pack_settings)
            main_profiler.end_section()
            
            main_profiler.section('augment_translate_new')
            a, b = image_util.augment_translate_new((packed_images, gt_images))
            main_profiler.end_section()
            
            main_profiler.section('augment_translate_old')
            a, b = image_util.augment_translate_old((packed_images, gt_images))
            main_profiler.end_section()
            
            main_profiler.section('augment_translate_old_loop')
            a, b = image_util.augment_translate_old_loop((packed_images, gt_images))
            main_profiler.end_section()
            
            main_profiler.section('augment_translate_fast')
            a, b = image_util.augment_translate_fast((packed_images, gt_images))
            main_profiler.end_section()
            
            main_profiler.section('augment_mirror')
            x = image_util.augment_mirror((packed_images, gt_images))
            main_profiler.end_section()

    main_profiler.print_evaluation()
    profiling.print_evaluation()