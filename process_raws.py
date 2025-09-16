import os
import rawpy
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import argparse
import options

if __name__ == '__main__':
    
    process_short = True
    enhance_short_exposure = True
    result_folder = './../processed/Sony/short_linear_enhance/'
    input_list_file = './data_lists/Sony_test_list_2.txt'
    
    
    os.makedirs(result_folder, exist_ok=True)

    opt = options.Options().init(argparse.ArgumentParser()).parse_args()
    
    with open(input_list_file) as fr:
        input_list = list(line.split(' ') for line in fr.readlines())
        
    for line in tqdm(input_list):
        in_name = line[0].strip()
        in_file = os.path.join(opt.dataset_folder, in_name[2:])
        in_filename = os.path.basename(in_name)
        in_exposure = float(in_filename[9:-5])
        
        gt_name = line[1].strip()
        gt_file = os.path.join(opt.dataset_folder, gt_name[2:])
        gt_filename = os.path.basename(gt_name)
        gt_exposure = float(gt_filename[9:-5])
        
        if process_short:
            exposure_factor = min(gt_exposure / in_exposure, 300) if enhance_short_exposure else 1
            processed = rawpy.imread(in_file).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            image = (processed * exposure_factor / 256.0).clip(0, 255).astype(np.uint8)
            if enhance_short_exposure:
                out_file = f"{in_filename[:-4]}_short_{int(exposure_factor)}.png"
            else:
                out_file = f"{in_filename[:-4]}_short.png"
        else:
            processed = rawpy.imread(gt_file).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            image = (processed / 256.0).clip(0, 255).as_type(np.uint8)
            out_file = f"{in_filename[:-4]}_long.png"
        
        Image.fromarray(image).save(os.path.join(result_folder, out_file))