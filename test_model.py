import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.sony_images import sid_original, sid_bottleneck_transformer, sid_no_bottleneck, dataset
from util import image_util
import cv2

import argparse
import options

if __name__ == '__main__':
    
    opt = options.Options().init(argparse.ArgumentParser()).parse_args()
    
    process_to_file = True
    save_gt = False
    
    result_folder = 'E:/workspace/Bach/Bach300/SID2/processed/Sony/sid_no_bottleneck/'
    
    print(f"Time now: {datetime.datetime.now().isoformat()}")

    # Model
    model = sid_no_bottleneck.Model()
    model.load_state()
    #model = sid_bottleneck_transformer.Model()
    #model.load_state('./models/sony_images/states/sid_bottleneck_transformer_initial.pt')
    

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    # ---------- DataLoader ----------
    dataset.preprocess_raw_gts(os.path.join(opt.dataset_folder, 'Sony', 'long'), opt.preprocess_folder)

    with open('./data_lists/Sony_test_list_2.txt') as fr:
        test_list = list(line.split(' ') for line in fr.readlines())
        
    dataset_test = dataset.RawImageDataset(test_list, opt.dataset_folder, opt.preprocess_folder, give_meta=True)
    
    dataloader_test = DataLoader(
        dataset_test, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        pin_memory=use_cuda, 
        drop_last=False, 
        persistent_workers=False
    )
    
    model.eval()
    total_psnr = 0
    time_begin = time.time()
        
    with torch.no_grad():
        for batch_idx, (in_images, gt_images, meta) in enumerate(tqdm(dataloader_test, f"Testing")):
            batch_size = in_images.shape[0]
            ids = meta['id']
            ratios = meta['ratio']
            
            in_images = in_images.to(device, non_blocking=True)
            gt_images = gt_images.to(device, non_blocking=True)
            
            out_images = model(in_images)
            out_images = out_images.clip(0.0, 1.0)
            
            psnr = image_util.batch_psnr(out_images, gt_images)
            total_psnr = psnr.mean().item() * batch_size
            
            gt_cv = image_util.tensor_to_images(gt_images, flip_to_bgr=not process_to_file)
            out_cv = image_util.tensor_to_images(out_images, flip_to_bgr=not process_to_file)
            
            for i in range(0, batch_size):
                if process_to_file:
                    os.makedirs(result_folder, exist_ok=True)
                    if save_gt:
                        Image.fromarray(gt_cv[i]).save(os.path.join(result_folder, f'{ids[i]}_{int(ratios[i])}_gt.png'))
                    Image.fromarray(out_cv[i]).save(os.path.join(result_folder, f'{ids[i]}_{int(ratios[i])}_out.png'))
                else:
                    im_gt = cv2.resize(gt_cv[i], None, fx=0.2, fy=0.2)
                    im_out = cv2.resize(out_cv[i], None, fx=0.2, fy=0.2)
                    cv2.imshow(f'{batch_idx*batch_size + i+1} {psnr[i]}', np.concatenate((im_gt, im_out), axis=1))
                    
                    # need to give cp some sleep time
                    cv2.waitKey()
                    cv2.destroyAllWindows()
    
    time_passed = time.time() - time_begin
    avg_psnr = total_psnr / len(dataset_test)