import os
import time
import datetime
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.sony_images import get_model_class, dataset
from util import image_util
import cv2

import argparse
import options

if __name__ == '__main__':
    
    opt = options.init_test(argparse.ArgumentParser()).parse_args()
    print(opt)
    
    model = get_model_class(opt.model)
    if opt.model_state is not None and opt.model_state != '':
        model_state = opt.model_state
        state_dict = torch.load(opt.model_state, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model_state = model.load_pretrained()
    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device {'cuda' if use_cuda else 'cpu'}.')
    
    model.to(device=device)
    if opt.compile_model:
        model = torch.compile(model)
        print('Model compile enabled.')
    
    if opt.auto_mixed_precision:
        print('Auto mixed precision enabled.')

    criterion = nn.L1Loss(reduction='none').to(device=device)

    # ---------- DataLoader ----------
    dataset.preprocess_raw_gts(os.path.join(opt.dataset_folder, 'Sony', 'long'), opt.preprocess_folder)
    
    with open(opt.test_list) as fr:
        test_list = list(line.split(' ') for line in fr.readlines())
    
    dataset_test = dataset.RawImageDataset(test_list, opt.dataset_folder, opt.preprocess_folder, give_meta=True, pack_augment_on_worker=False)
    dataloader_batch_size = opt.validation_batch_size
    dataloader_test = DataLoader(
        dataset_test, 
        batch_size=dataloader_batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        pin_memory=use_cuda, 
        drop_last=False, 
        persistent_workers=False
    )
    
    model.eval()
    
    losses = torch.empty((len(dataset_test)), dtype=torch.float32, device=device)
    psnrs = torch.empty((len(dataset_test)), dtype=torch.float32, device=device)
    
    ids = []
    ratios = []
    
    os.makedirs(opt.out_folder, exist_ok=True)
    
    time_begin = time.time()
        
    with torch.no_grad():
        for batch_idx, ((raw_images, pack_settings), gt_images, meta) in enumerate(tqdm(dataloader_test, f"Testing")):
            batch_size = raw_images.shape[0]
            
            raw_images = raw_images.to(device, non_blocking=True)
            pack_settings = {key: value.to(device, non_blocking=True) for key, value in pack_settings.items()}
            gt_images = gt_images.to(device, non_blocking=True, dtype=torch.float32)
            
            packed = dataset.pack_raw(raw_images, pack_settings)
            if dataset_test.transform is not None:
                packed, gt_images = dataset_test.transform((packed, gt_images))
            
            if opt.auto_mixed_precision:
                with torch.amp.autocast(device.type):
                    out_images = model(packed)
            else:
                out_images = model(packed)
            
            out_images = out_images.clip(0.0, 1.0)
            
            loss = criterion(out_images, gt_images).mean(dim=[1, 2, 3])
            psnr = image_util.batch_psnr(out_images, gt_images)
            
            index = batch_idx*dataloader_batch_size
            losses[index:index + batch_size] = loss
            psnrs[index:index + batch_size] = psnr
            
            for i in range(batch_size):
                ids.append(meta['id'][i])
                ratios.append(meta['ratio'][i].item())
            
            gt_np = image_util.tensor_to_images(gt_images)
            out_np = image_util.tensor_to_images(out_images)
            
            if opt.save_images:
                for i in range(batch_size):
                    Image.fromarray(out_np[i]).save(os.path.join(opt.out_folder, f'{meta['id'][i]}_{int(meta['ratio'][i])}_out.png'))
            if opt.show_images:
                gt_np = image_util.images_flip_rgb_bgr(gt_np)
                out_np = image_util.images_flip_rgb_bgr(out_np)
                for i in range(batch_size):
                    im_gt = cv2.resize(gt_np[i], None, fx=0.2, fy=0.2)
                    im_out = cv2.resize(out_np[i], None, fx=0.2, fy=0.2)
                image_util.show_image(f'{batch_idx*batch_size + i+1} {psnr[i]}', np.concatenate((im_gt, im_out), axis=1))
                
                # need to give cp some sleep time
                cv2.waitKey()
                cv2.destroyAllWindows()
    
    psnrs = psnrs.cpu()
    losses = losses.cpu()
    
    results = {
        'total_time': time.time() - time_begin,
        'model': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'model_state': model_state,
        'avg_psnr': psnrs.mean().item(),
        'avg_loss': losses.mean().item(),
        'losses': losses.tolist(),
        'psnrs': psnrs.tolist(),
        'ids': ids,
        'ratios': ratios
    }
    
    with open(os.path.join(opt.out_folder, 'results.json'), 'w') as fw:
        fw.write(json.dumps(results))
    
    print(f"Average PSNR: {results['avg_psnr']}, Average loss {results['avg_loss']}")