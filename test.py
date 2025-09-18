from models.sony_images import dataset
import argparse
import options
import os
import torch
from torch.utils.data import DataLoader
from util import profiler
from tqdm import tqdm

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = options.Options().init(argparse.ArgumentParser()).parse_args()
        
    if opt.preload_gts:
        gt_data = dataset.GTDict(os.path.join(opt.dataset_folder, 'Sony', 'long'))
    else:
        dataset.preprocess_raw_gts(os.path.join(opt.dataset_folder, 'Sony', 'long'), opt.preprocess_folder)
        gt_data = opt.preprocess_folder

    with open('./data_lists/Sony_val_list.txt') as fr:
        val_list = list(line.split(' ') for line in fr.readlines())

    dataset_val = dataset.RawImageDataset(val_list, opt.dataset_folder, gt_data, pack_augment_on_worker=False)

    dataloader = DataLoader(
        dataset_val, 
        batch_size=opt.validation_batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        pin_memory=use_cuda, 
        drop_last=False, 
        persistent_workers=not dataset_val.pack_augment_on_worker
    )

    profiler = profiler.Profiler()

    for ((raw_images, pack_settings), gt_images) in tqdm(dataloader):
        
        profiler.section("promote 0")
        
        raw_images = raw_images.to(device=device, dtype=torch.float32, non_blocking=True)
        pack_settings = {key: value.to(device=device, non_blocking=True) for key, value in pack_settings.items()}
        gt_images = gt_images.to(device=device, dtype=torch.float32, non_blocking=True)
        
        packed = dataset.pack_raw(raw_images, pack_settings)
        
        if use_cuda:
            torch.cuda.synchronize(device)
        profiler.end_section()
        
        profiler.section("promote 1")
        
        raw_images = raw_images.to(device=device, non_blocking=True)
        pack_settings = {key: value.to(device=device, non_blocking=True) for key, value in pack_settings.items()}
        gt_images = gt_images.to(device=device, dtype=torch.float32, non_blocking=True)
        
        packed = dataset.pack_raw_promote_1(raw_images, pack_settings)
        
        if use_cuda:
            torch.cuda.synchronize(device)
        profiler.end_section()
        
        profiler.section("promote 2")
        
        raw_images = raw_images.to(device=device, non_blocking=True)
        pack_settings = {key: value.to(device=device, non_blocking=True) for key, value in pack_settings.items()}
        gt_images = gt_images.to(device=device, dtype=torch.float32, non_blocking=True)
        
        packed = dataset.pack_raw_promote_2(raw_images, pack_settings)
        
        if use_cuda:
            torch.cuda.synchronize(device)
        profiler.end_section()

    profiler.print_evaluation()