import random
import io
import os
import time
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from models.sony_images import sid_bottleneck_transformer, dataset
from util import image_util, file_storage, profiler

import argparse
import options

if __name__ == '__main__':
    opt = options.Options().init(argparse.ArgumentParser()).parse_args()
    
    print(f"Time now: {datetime.datetime.now().isoformat()}")
    print(f"CPU core count is {os.cpu_count()}")
    print(opt)

    if opt.use_s3_storage:
        userdata = file_storage.JsonUserdata('./s3_access.json')
        object_storage = file_storage.S3ObjectStorage(userdata)

    # enable cuDNN benchmark mode to speed up training
    torch.backends.cudnn.benchmark = True

    # set seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Model
    model = sid_bottleneck_transformer.Model()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {'cuda' if torch.cuda.is_available() else 'cpu'}.')
    
    # Optimizer
    encoder_params = []
    bottleneck_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if name.startswith('conv1_') or name.startswith('conv2_') or name.startswith('conv3_') or name.startswith('conv4_'):
            encoder_params.append(param)
        elif name.startswith('bottleneck5'):
            bottleneck_params.append(param)
        elif name.startswith('up') or name.startswith('conv'):
            decoder_params.append(param)
        else:
            raise RuntimeError(f'Unaccounted model parameters: {name}')
    
    optimizer_params = []
    optimizer_param_group_indices = [-1] * 3

    if opt.encoder_initial_lr > 0:
        optimizer_param_group_indices[0] = len(optimizer_params)
        optimizer_params.append({
            'params': encoder_params,
            'lr': opt.encoder_initial_lr,
            'weight_decay': opt.encoder_weight_decay
        })
    else:
        for param in encoder_params:
            param.requires_grad = False
        print('Encoder frozen.')
    
    
    if opt.bottleneck_initial_lr > 0:
        optimizer_param_group_indices[1] = len(optimizer_params)
        optimizer_params.append({
                'params': bottleneck_params,
                'lr': opt.bottleneck_initial_lr,
                'weight_decay': opt.bottleneck_weight_decay
            })
    else:
        for param in bottleneck_params:
            param.requires_grad = False
        print('Bottleneck frozen.')
    
    if opt.decoder_initial_lr > 0:
        optimizer_param_group_indices[2] = len(optimizer_params)
        optimizer_params.append({
                'params': decoder_params,
                'lr': opt.decoder_initial_lr,
                'weight_decay': opt.decoder_weight_decay
            })
    else:
        for param in decoder_params:
            param.requires_grad = False
        print('Decoder frozen')
    
    optimizer = torch.optim.AdamW(optimizer_params)
    if opt.auto_mixed_precision:
        print('Auto mixed precision enabled.')
        scaler = torch.amp.GradScaler()

    # Resume
    if opt.resume_epoch != 0:
        
        # load log and model from s3 or storage
        start_epoch = opt.resume_epoch
        if opt.use_s3_storage:
            log_str = object_storage.load_file(opt.s3_prefix + f'log_{opt.resume_epoch}.json', binary=False)
            log = json.loads(log_str)
            
            buffer = object_storage.load_file(opt.s3_prefix + f'model_checkpoint_{opt.resume_epoch}.pt', binary=True)
            model_checkpoint = torch.load(buffer, weights_only=True)
        else:
            with open(os.path.join(opt.out_path, f'log_{opt.resume_epoch}.json'), 'r') as fr:
                log = json.load(fr)
            
            model_file = os.path.join(opt.out_path, f'model_checkpoint_{opt.resume_epoch}.pt')
            model_checkpoint = torch.load(model_file, weights_only=False)
        
        model.load_state_dict(model_checkpoint)
        print(f'Loaded model_checkpoint_{opt.resume_epoch}')
        
        if opt.load_optimizer:
            # also load in what epoch lr schedule started
            lr_schedule_first_epoch = log['lr_schedule_first_epoch']
            
            if opt.use_s3_storage:
                buffer = object_storage.load_file(opt.s3_prefix + f'optimizer_checkpoint_{opt.resume_epoch}.pt', binary=True)
                optimizer_checkpoint = torch.load(buffer, weights_only=True)
            else:
                optimizer_file = os.path.join(opt.out_path, f'optimizer_checkpoint_{opt.resume_epoch}.pt')
                optimizer_checkpoint = torch.load(optimizer_file, weights_only=False)
                
            optimizer.load_state_dict(optimizer_checkpoint)
            print(f'Loaded optimizer_checkpoint_{opt.resume_epoch}.pt')
        else:
            # Restart warmup
            lr_schedule_first_epoch = start_epoch
            
        print(f'Resuming with epoch {start_epoch+1}.')
        
    else:
        start_epoch = 0
        lr_schedule_first_epoch = 0
        model.load_state('./models/sony_images/states/sid_bottleneck_transformer_initial.pt')
        
        print(f'Starting in epoch 1.')
    
    model.to(device=device)
    if opt.compile_model:
        model = torch.compile(model)
        print('Model compile enabled')

    # Scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=opt.warmup_epochs,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opt.total_epochs - opt.warmup_epochs,
        eta_min=1e-6
    )
    
    scheduler = SequentialLR(
        optimizer,
        last_epoch=start_epoch - lr_schedule_first_epoch,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[opt.warmup_epochs]
    )

    # Loss
    criterion = nn.L1Loss().to(device=device)

    # ---------- DataLoader ----------
    
    if opt.preload_gts:
        gt_data = dataset.GTDict(os.path.join(opt.dataset_folder, 'Sony', 'long'))
    else:
        dataset.preprocess_raw_gts(os.path.join(opt.dataset_folder, 'Sony', 'long'), opt.preprocess_folder, 4)
        dt_data = opt.preprocess_folder
        
    with open('./data_lists/Sony_train_list.txt') as fr:
        train_list = list(line.split(' ') for line in fr.readlines())
    with open('./data_lists/Sony_val_list.txt') as fr:
        val_list = list(line.split(' ') for line in fr.readlines())
    
    dataset_train = dataset.RawImageDataset(train_list, opt.dataset_folder, gt_data, pack_augment_on_worker=False)
    dataset_val = dataset.RawImageDataset(val_list, opt.dataset_folder, gt_data, pack_augment_on_worker=False)

    len_train_set = len(dataset_train)
    len_val_set = len(dataset_val)
    
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=opt.num_workers, 
        pin_memory=use_cuda, 
        drop_last=True, 
        persistent_workers=not dataset_train.pack_augment_on_worker
    )

    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=opt.validation_batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        pin_memory=use_cuda, 
        drop_last=False, 
        persistent_workers=not dataset_val.pack_augment_on_worker
    )

    print(f"{len_train_set} training images, {len_val_set} validation images")

    assert opt.effective_batch_size % opt.batch_size == 0
    gradient_acc_total_steps = opt.effective_batch_size // opt.batch_size
    
    profiler = profiler.Profiler()

    for epoch_idx in range(start_epoch, opt.total_epochs):
        epoch_number = epoch_idx + 1
        
        if epoch_number >= opt.augment_images_epoch:
            dataloader_train.transform = image_util.AugmentSequentiel(
                image_util.AugmentTranslateReflect(max_translate_factor=0.4, chance=0.75),
                image_util.augment_mirror
            )
            augment_images = True
        else:
            dataloader_train.transform = None
            augment_images = False
        
        log = {}
        log['epoch'] = epoch_number
        log['lr_schedule_first_epoch'] = lr_schedule_first_epoch
        log['learning_rates'] = [optimizer.param_groups[optimizer_param_group_indices[i]]['lr'] if optimizer_param_group_indices[i] != -1 else 0 for i in range(3)]
        log['auto_mixed_precision'] = opt.auto_mixed_precision
        log['augment_images'] = augment_images
        
        # ---------- train ----------
        
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        gradient_acc = 0
        
        time_begin = time.time()
        
        profiler.section('wait data (train)')
        for batch_idx, ((raw_images, pack_settings), gt_images) in enumerate(tqdm(dataloader_train, f"Training epoch {epoch_number}")):
            profiler.end_section()
            batch_size = raw_images.shape[0]
            
            raw_images = raw_images.to(device, non_blocking=True)
            pack_settings = {key: value.to(device, non_blocking=True) for key, value in pack_settings.items()}
            gt_images = gt_images.to(device, non_blocking=True)
            
            with torch.no_grad():
                profiler.section('pack (train)')
                packed = dataset.pack_raw(raw_images, pack_settings)
                profiler.end_section()
                
                if dataset_train.transform is not None:
                    profiler.section('augment (train)')
                    packed, gt_images = dataset_train.transform((packed, gt_images))
                    profiler.end_section()
            
            profiler.section('model (train)')
            if opt.auto_mixed_precision:
                with torch.amp.autocast(device.type):
                    out_images = model(packed)
                    loss = criterion(out_images, gt_images)
            else:
                out_images = model(packed)
                loss = criterion(out_images, gt_images)
            profiler.end_section()
                
                
            total_loss += loss.item() * batch_size
            loss = loss / gradient_acc_total_steps
            
            profiler.section('backward loss (train)')
            if opt.auto_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            profiler.end_section()
            
            gradient_acc += 1
            if gradient_acc == gradient_acc_total_steps:
                # update weights and reset gradients
                profiler.section('optimize (train)')
                if opt.auto_mixed_precision:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
    
                optimizer.zero_grad()
                gradient_acc = 0
                profiler.end_section()
            
            profiler.section('wait data (train)')
        
        profiler.end_section()
        
        # handle accumulated gradients after last update
        if gradient_acc != 0:
            if opt.auto_mixed_precision:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
        
        # update LR scheduler
        scheduler.step()
        
        log['avg_train_loss'] = total_loss / len_train_set
        log['train_time'] = time.time() - time_begin
        
        # ---------- validate ----------
        model.eval()
        total_loss = 0
        total_psnr = 0
        time_begin = time.time()
            
        with torch.no_grad():
            profiler.section('wait data (val)')
            for batch_idx, ((raw_images, pack_settings), gt_images) in enumerate(tqdm(dataloader_val, f"Validation epoch {epoch_number}")):
                profiler.end_section()
                batch_size = raw_images.shape[0]
                
                raw_images = raw_images.to(device, non_blocking=True)
                pack_settings = {key: value.to(device, non_blocking=True) for key, value in pack_settings.items()}
                gt_images = gt_images.to(device, non_blocking=True)
                
                with torch.no_grad():
                    profiler.section('pack (val)')
                    packed = dataset.pack_raw(raw_images, pack_settings)
                    profiler.end_section()
                
                profiler.section('model (val)')
                out_images = model(packed)
                out_images = out_images.clip(0.0, 1.0)
                
                loss = criterion(out_images, gt_images)
                total_loss += loss.item() * batch_size
                total_psnr += image_util.batch_psnr(out_images, gt_images).mean().item() * batch_size
                profiler.end_section()

                profiler.section('wait data (val)')
            profiler.end_section()
        
        profiler.print_evaluation()
        profiler.reset_timings()
        
        log['avg_val_loss'] = total_loss / len_val_set
        log['avg_val_psnr'] = total_psnr / len_val_set
        log['val_time'] = time.time() - time_begin
        
        # store logs and checkpoints
        if opt.use_s3_storage:
            object_storage.store_file(opt.s3_prefix + f'log_{epoch_number}.json', json.dumps(log), binary=False)
            
            if epoch_number % opt.save_checkpoint_frequency == 0:
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                object_storage.store_file(opt.s3_prefix + f'model_checkpoint_{epoch_number}.pt', buffer, binary=True)
                
                buffer = io.BytesIO()
                torch.save(optimizer.state_dict(), buffer)
                buffer.seek(0)
                object_storage.store_file(opt.s3_prefix + f'optimizer_checkpoint_{epoch_number}.pt', buffer, binary=True)
        else:
            os.makedirs(opt.out_path, exist_ok=True)
            with open(os.path.join(opt.out_path, f'log_{epoch_number}.json'), 'w')as fr:
                fr.write(json.dumps(log))
            
            if epoch_number % opt.save_checkpoint_frequency == 0:
                torch.save(model.state_dict(), os.path.join(opt.out_path, f'model_checkpoint_{epoch_number}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(opt.out_path, f'optimizer_checkpoint_{epoch_number}.pt'))
        
        print(f'Epoch {epoch_number}: Train loss {log['avg_train_loss']}, Validation loss {log['avg_val_loss']}, Validation PSNR {log['avg_val_psnr']}')