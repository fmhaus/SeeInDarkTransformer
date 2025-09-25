import random
import io
import os
import time
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from models.sony_images import dataset, sid_bottleneck_transformer, get_model_class
from util import image_util
import config

def init_options(parser):
    config.init_common_args(parser)
    
    # train parameters
    parser.add_argument('--resume_epoch', type=int, default=0, help='epoch to resume training from (0 = train from zero)')
    parser.add_argument('--total_epochs', type=int, default=200, help='toal number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs (0 = no warmup)')
    parser.add_argument('--augment_images_epoch', type=int, default=5, help='After what epoch images should be augmented (with random crops and flips)')
    parser.add_argument('--load_optimizer', action='store_true', default=False, help='Whether to load the optimizer (and lr schedule) when using resume or not')
    parser.add_argument('--save_checkpoint_frequency', type=int, default=1, help='After how many epochs the model and optimizer checkpoints should be saved')
    parser.add_argument('--effective_batch_size', type=int, default=24, help='number of images processed before updating weights')
    parser.add_argument('--encoder_initial_lr', type=float, default=1e-4, help='Initial learning rate for encoder (0: encoder frozen)')
    parser.add_argument('--bottleneck_initial_lr', type=float, default=1e-4, help='Initial learning rate for bottleneck (0: bottleneck frozen)')
    parser.add_argument('--decoder_initial_lr', type=float, default=1e-4, help='Initial learning rate for decoder (0: decoder frozen)')
    parser.add_argument('--encoder_weight_decay', type=float, default=0, help='Weight decay for encoder')
    parser.add_argument('--bottleneck_weight_decay', type=float, default=0, help='Weight decay for bottleneck')
    parser.add_argument('--decoder_weight_decay', type=float, default=0, help='Weight decay for decoder')
    parser.add_argument('--mlp_dropout', type=float, default=0.1, help='Dropout value for the Transformer MLP')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='Dropout value for the Transformer Attention')
       
    return parser

if __name__ == '__main__':
    
    opt = init_options(argparse.ArgumentParser()).parse_args()
    device_cfg = config.DeviceConfig(opt.device_config)
    
    print(f"Time now: {datetime.datetime.now().isoformat()}")
    print(f"CPU core count is {os.cpu_count()}.")
    print(opt)

    # set seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    
    # device setup
    if device_cfg.compile_model:
        torch.backends.cudnn.benchmark = True
        
    device = torch.device('cuda' if device_cfg.use_cuda else 'cpu')
    print(f'Using device {'cuda' if device_cfg.use_cuda else 'cpu'}.')
    
    best_psnr = 0.0

    # Resume
    if opt.resume_epoch != 0:
        
        # load log and model
        start_epoch = opt.resume_epoch
        
        with open(os.path.join(opt.out_folder, f'log_{opt.resume_epoch}.json'), 'r') as fr:
            log = json.load(fr)
        
        model_file = os.path.join(opt.out_folder, f'model_checkpoint_{opt.resume_epoch}.pt')
        model_checkpoint = torch.load(model_file, map_location=device)
        
        print(f'Loaded model_checkpoint_{opt.resume_epoch}.pt')
        
        if opt.load_optimizer:
            # also load in what epoch lr schedule started
            lr_schedule_first_epoch = log['lr_schedule_first_epoch']
           
            optimizer_file = os.path.join(opt.out_folder, f'optimizer_checkpoint_{opt.resume_epoch}.pt')
            optimizer_checkpoint = torch.load(optimizer_file, map_location=device)
                
            print(f'Loaded optimizer_checkpoint_{opt.resume_epoch}.pt.')
        else:
            # Restart warmup
            lr_schedule_first_epoch = start_epoch
            optimizer_checkpoint = None
        
        best_log_file = os.path.join(opt.out_folder, f'log_best.json')
        if os.path.isfile(best_log_file):
            with open(best_log_file, 'r') as fr:
                best_log = json.load(fr)
                best_psnr = best_log['avg_val_psnr']
        
        print(f'Resuming with epoch {start_epoch+1}.')
        
    else:
        start_epoch = 0
        lr_schedule_first_epoch = 0
        model_checkpoint = torch.load('./models/sony_images/states/sid_bottleneck_transformer_initial_2b_c.pt', map_location=device)
        optimizer_checkpoint = None
        
        print(f'Starting in epoch 1.')

    # Model 
    model_class = get_model_class(opt.model)
    model = model_class()
    model.load_state_dict(model_checkpoint)
    
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
        for param in encoder_params:
            param.requires_grad = True
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
        for param in bottleneck_params:
            param.requires_grad = True
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
        for param in decoder_params:
            param.requires_grad = True
    else:
        for param in decoder_params:
            param.requires_grad = False
        print('Decoder frozen')
    
    optimizer = torch.optim.AdamW(optimizer_params)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint)
        
    if device_cfg.auto_mixed_precision:
        print('Auto mixed precision enabled.')
        scaler = torch.amp.GradScaler()

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
        milestones=[opt.warmup_epochs+1]
    )

    print(f'Starting LR schedule on epoch {start_epoch - lr_schedule_first_epoch + 1}.')

    model.set_transformer_dropout(opt.attn_dropout, opt.mlp_dropout)
    
    model_uncompiled = model
    if device_cfg.compile_model:
        model = torch.compile(model)
        print('Model compile enabled.')

    # Loss
    criterion = nn.L1Loss().to(device=device)

    # ---------- DataLoader ----------
    
    dataset.preprocess_raw_gts(os.path.join(opt.dataset_folder, 'Sony', 'long'), opt.preprocess_folder, device_cfg.num_workers)
    
    if device_cfg.preload_gts:
        gt_data = dataset.GTDict(opt.preprocess_folder, device_cfg.num_workers)
    else:
        gt_data = opt.preprocess_folder
        
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
        batch_size=device_cfg.train_batch_size, 
        shuffle=True, 
        num_workers=device_cfg.num_workers, 
        pin_memory=device_cfg.use_cuda, 
        drop_last=True, 
        persistent_workers=not dataset_train.pack_augment_on_worker
    )

    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=device_cfg.validation_batch_size, 
        shuffle=False, 
        num_workers=device_cfg.num_workers, 
        pin_memory=device_cfg.use_cuda, 
        drop_last=False, 
        persistent_workers=not dataset_val.pack_augment_on_worker
    )

    os.makedirs(opt.out_folder, exist_ok=True)
    print(f"{len_train_set} training images, {len_val_set} validation images.")

    assert opt.effective_batch_size % device_cfg.train_batch_size == 0
    gradient_acc_total_steps = opt.effective_batch_size // device_cfg.train_batch_size

    with open(os.path.join(opt.out_folder, 'options.json'), 'w') as fr:
        fr.write(str(opt))

    for epoch_idx in range(start_epoch, opt.total_epochs):
        epoch_number = epoch_idx + 1
        
        if epoch_number >= opt.augment_images_epoch:
            dataloader_train.transform = image_util.AugmentSequentiel(
                image_util.AugmentTranslateReflect(max_translate_factor=0.5, chance=0.95),
                image_util.augment_mirror
            )
            augment_images = True
        else:
            dataloader_train.transform = None
            augment_images = False
        
        log = {}
        log['model'] = f"{model_uncompiled.__class__.__module__}.{model_uncompiled.__class__.__name__}"
        log['epoch'] = epoch_number
        log['lr_schedule_first_epoch'] = lr_schedule_first_epoch
        log['learning_rates'] = [optimizer.param_groups[optimizer_param_group_indices[i]]['lr'] if optimizer_param_group_indices[i] != -1 else 0 for i in range(3)]
        log['auto_mixed_precision'] = device_cfg.auto_mixed_precision
        log['augment_images'] = augment_images
        
        # ---------- train ----------
        
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        gradient_acc = 0
        
        time_begin = time.time()
        
        for batch_idx, ((raw_images, pack_settings), gt_images) in enumerate(tqdm(dataloader_train, f"Training epoch {epoch_number}")):
            batch_size = raw_images.shape[0]
            
            raw_images = raw_images.to(device, non_blocking=True)
            pack_settings = {key: value.to(device, non_blocking=True) for key, value in pack_settings.items()}
            gt_images = gt_images.to(device, non_blocking=True)
            
            with torch.no_grad():
                packed = dataset.pack_raw(raw_images, pack_settings)
                
                if dataset_train.transform is not None:
                    packed, gt_images = dataset_train.transform((packed, gt_images))
            
            if device_cfg.auto_mixed_precision:
                with torch.amp.autocast(device.type):
                    out_images = model(packed)
                    loss = criterion(out_images, gt_images)
            else:
                out_images = model(packed)
                loss = criterion(out_images, gt_images)
                
                
            total_loss += loss.item() * batch_size
            loss = loss / gradient_acc_total_steps
            
            if device_cfg.auto_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            gradient_acc += 1
            if gradient_acc == gradient_acc_total_steps:
                # update weights and reset gradients
                if device_cfg.auto_mixed_precision:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
    
                optimizer.zero_grad()
                gradient_acc = 0
        
        # handle accumulated gradients after last update
        if gradient_acc != 0:
            if device_cfg.auto_mixed_precision:
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
            for batch_idx, ((raw_images, pack_settings), gt_images) in enumerate(tqdm(dataloader_val, f"Validation epoch {epoch_number}")):
                batch_size = raw_images.shape[0]
                
                raw_images = raw_images.to(device, non_blocking=True)
                pack_settings = {key: value.to(device, non_blocking=True) for key, value in pack_settings.items()}
                gt_images = gt_images.to(device, non_blocking=True)
                
                packed = dataset.pack_raw(raw_images, pack_settings)
                
                out_images = model(packed)
                out_images = out_images.clip(0.0, 1.0)
                
                loss = criterion(out_images, gt_images)
                total_loss += loss.item() * batch_size
                total_psnr += image_util.batch_psnr(out_images, gt_images).mean().item() * batch_size
        
        log['avg_val_loss'] = total_loss / len_val_set
        log['avg_val_psnr'] = total_psnr / len_val_set
        log['val_time'] = time.time() - time_begin
        
        print(f'Epoch {epoch_number}: Train loss {log['avg_train_loss']}, Validation loss {log['avg_val_loss']}, Validation PSNR {log['avg_val_psnr']}')
        
        # store logs and checkpoints
        
        with open(os.path.join(opt.out_folder, f'log_{epoch_number}.json'), 'w')as fr:
            fr.write(json.dumps(log))
        
        if epoch_number % opt.save_checkpoint_frequency == 0:
            torch.save(model_uncompiled.state_dict(), os.path.join(opt.out_folder, f'model_checkpoint_{epoch_number}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(opt.out_folder, f'optimizer_checkpoint_{epoch_number}.pt'))
            print('Saved checkpoint.')
        
        if log['avg_val_psnr'] > best_psnr:
            best_psnr = log['avg_val_psnr']
            with open(os.path.join(opt.out_folder, f'log_best.json'), 'w')as fr:
                fr.write(json.dumps(log))
            torch.save(model_uncompiled.state_dict(), os.path.join(opt.out_folder, f'model_checkpoint_best.pt'))
            torch.save(optimizer.state_dict(), os.path.join(opt.out_folder, f'optimizer_checkpoint_best.pt'))
            print('Saved new best.')