import torch
import torch.nn.functional as F
import numpy as np
import os
import rawpy
from tqdm import tqdm

def preprocess_raw_gts(gt_folder, preprocessed_folder):
    os.makedirs(preprocessed_folder, exist_ok=True)
    
    for filename in tqdm(os.listdir(gt_folder), 'Preprocessing ground truth images'):
        npy_file = os.path.join(preprocessed_folder, filename + '.npy')
        if not os.path.exists(npy_file):
            raw_file = os.path.join(gt_folder, filename)
            gt = rawpy.imread(raw_file).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            tensor = np.permute_dims(np.float32(gt / 65535.0), axes=(2, 0, 1))
            np.save(npy_file, tensor)

def pack_raw(raw, ratio):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)
    
    im = np.expand_dims(im, axis=0)
    img_shape = im.shape
    H = img_shape[1]
    W = img_shape[2]
         
    out = np.concatenate((im[:, 0:H:2, 0:W:2],
                            im[:, 0:H:2, 1:W:2],
                            im[:, 1:H:2, 1:W:2],
                            im[:, 1:H:2, 0:W:2]), axis=0)
        
    out = np.minimum(out * ratio, 1.0)
    
    return torch.from_numpy(out)

class RawImageDataset(torch.utils.data.Dataset):
    def __init__(self, set_list, dataset_folder, preprocessed_gt_folder, give_meta=False):
        self.items = set_list
        self.dataset_folder = dataset_folder
        self.gt_folder = preprocessed_gt_folder
        self.transform = None
        self.give_meta = give_meta
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        in_name = self.items[idx][0].strip()
        in_file = os.path.join(self.dataset_folder, in_name[2:])
        in_filename = os.path.basename(in_name)
        in_exposure = float(in_filename[9:-5])
        
        gt_name = self.items[idx][1].strip()
        gt_filename = os.path.basename(gt_name)
        gt_exposure = float(gt_filename[9:-5])
        
        raw = rawpy.imread(in_file)
        ratio = min(300, gt_exposure / in_exposure)
        in_image = pack_raw(raw, ratio)
        
        gt_file = os.path.join(self.gt_folder, gt_filename + '.npy')
        gt_image = torch.from_numpy(np.load(gt_file, allow_pickle=True))
        
        if self.transform is not None:
            in_image, gt_image = self.transform((in_image, gt_image))
        
        if self.give_meta:
            return in_image, gt_image, {'id': in_filename[:8], 'ratio': ratio}
        else:
            return in_image, gt_image