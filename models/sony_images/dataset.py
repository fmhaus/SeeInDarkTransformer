import torch
import numpy as np
import os
import rawpy
from tqdm import tqdm
from util.profiler import Profiler, DummyProfiler

class GTDict():
    def __init__(self, gt_folder):
        self.tensors = {}
        for filename in tqdm(os.listdir(gt_folder), 'Preloading GTs'):
            raw_file = os.path.join(gt_folder, filename)
            gt = rawpy.imread(raw_file).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            tensor = np.permute_dims(np.float16(gt / 65535.0), axes=(2, 0, 1))
            self.tensors[filename] = tensor
    
    def get(self, filename):
        return self.tensors[filename]
            

def preprocess_raw_gts(gt_folder, preprocess_folder):
    os.makedirs(preprocess_folder, exist_ok=True)
    for filename in os.listdir(gt_folder, 'Preprocessing GTs'):
        npy_file = os.path.join(preprocess_folder, filename + '.npy')
        if not os.path.exists(npy_file):
            raw_file = os.path.join(gt_folder, filename)
            gt = rawpy.imread(raw_file).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            tensor = np.permute_dims(np.float16(gt / 65535.0), axes=(2, 0, 1))
            np.save(npy_file, tensor)

def get_pack_settings(raw, exposure_ratio):
    return {
        'black_level': torch.tensor(raw.black_level_per_channel, dtype=torch.float32),
        'white_level': torch.tensor(raw.white_level, dtype=torch.float32).expand(4),
        'exposure_ratio': torch.tensor(exposure_ratio, dtype=torch.float32).clamp(max=300)
    }

def pack_raw(raw_image, settings):
    N, H_2, W_2 = raw_image.shape
    H = H_2 // 2
    W = W_2 // 2
    
    packed = torch.empty((N, 4, H, W), dtype=raw_image.dtype, device=raw_image.device)
    packed[:, 0] = raw_image[:, 0::2, 0::2]
    packed[:, 1] = raw_image[:, 0::2, 1::2]
    packed[:, 2] = raw_image[:, 1::2, 1::2]
    packed[:, 3] = raw_image[:, 1::2, 0::2]
    
    # match black and white level to [., C, ., .]
    black_level = settings['black_level'].view(N, 4, 1, 1)
    white_level = settings['white_level'].view(N, 4, 1, 1)
    exposure_ratio = settings['exposure_ratio'].view(N, 1, 1, 1)
    
    normalized = (packed - black_level).clamp(min=0.0) / (white_level - black_level)
    scaled = normalized * exposure_ratio
    
    return scaled.clamp(max=1.0)

def pack_raw_promote_1(raw_image, settings):
    N, H_2, W_2 = raw_image.shape
    H = H_2 // 2
    W = W_2 // 2
    
    packed = torch.empty((N, 4, H, W), dtype=torch.float32, device=raw_image.device)
    packed[:, 0] = raw_image[:, 0::2, 0::2].to(torch.float32)
    packed[:, 1] = raw_image[:, 0::2, 1::2].to(torch.float32)
    packed[:, 2] = raw_image[:, 1::2, 1::2].to(torch.float32)
    packed[:, 3] = raw_image[:, 1::2, 0::2].to(torch.float32)
    
    # match black and white level to [., C, ., .]
    black_level = settings['black_level'].view(N, 4, 1, 1)
    white_level = settings['white_level'].view(N, 4, 1, 1)
    exposure_ratio = settings['exposure_ratio'].view(N, 1, 1, 1)
    
    normalized = (packed - black_level).clamp(min=0.0) / (white_level - black_level)
    scaled = normalized * exposure_ratio
    
    return scaled.clamp(max=1.0)

def pack_raw_promote_2(raw_image, settings):
    N, H_2, W_2 = raw_image.shape
    H = H_2 // 2
    W = W_2 // 2
    
    packed = torch.empty((N, 4, H, W), dtype=raw_image.dtype, device=raw_image.device)
    packed[:, 0] = raw_image[:, 0::2, 0::2]
    packed[:, 1] = raw_image[:, 0::2, 1::2]
    packed[:, 2] = raw_image[:, 1::2, 1::2]
    packed[:, 3] = raw_image[:, 1::2, 0::2]
    
    # match black and white level to [., C, ., .]
    black_level = settings['black_level'].view(N, 4, 1, 1)
    white_level = settings['white_level'].view(N, 4, 1, 1)
    exposure_ratio = settings['exposure_ratio'].view(N, 1, 1, 1)
    
    normalized = (packed.to(torch.float32) - black_level).clamp(min=0.0) / (white_level - black_level)
    scaled = normalized * exposure_ratio
    
    return scaled.clamp(max=1.0)
    

class RawImageDataset(torch.utils.data.Dataset):
    def __init__(self, set_list, dataset_folder, gt_data, give_meta=False, pack_augment_on_worker=True, profiling=None):
        self.items = set_list
        self.dataset_folder = dataset_folder
        if isinstance(gt_data, str):
            self.gt_folder = gt_data
            self.preloaded_gt = False
        elif isinstance(gt_data, GTDict):
            self.gt_dict = gt_data
            self.preloaded_gt = True
        self.transform = None
        self.give_meta = give_meta
        self.pack_augment_on_worker = pack_augment_on_worker
        self.profiling = profiling
    
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
        
        profiler = Profiler() if self.profiling is not None else DummyProfiler()
        
        profiler.section('read_raw')
        raw = rawpy.imread(in_file)
        image_raw = torch.tensor(raw.raw_image_visible, dtype=torch.float16)
        profiler.end_section()
        
        profiler.section('load_gt')
        if self.preloaded_gt:
            gt_image = self.gt_dict.get(gt_filename)
        else:
            gt_file = os.path.join(self.gt_folder, gt_filename + '.npy')
            gt_image = torch.from_numpy(np.load(gt_file, allow_pickle=True))
        profiler.end_section()
        
        pack_settings = get_pack_settings(raw, gt_exposure / in_exposure)
        
        if self.pack_augment_on_worker:
            with torch.no_grad():
                profiler.section('pack_raw')
                # pack_raw need batch dimension, unsqueeze before
                image_raw = image_raw.unsqueeze(0)
                input = pack_raw(image_raw, pack_settings)
                profiler.end_section()
                
                profiler.section('augment')
                if self.transform is not None:
                    gt_image = gt_image.unsqueeze(0)
                    input, gt_image = self.transform((input, gt_image))
                    gt_image = gt_image.squeeze(0)
                    
                input = input.squeeze(0)
                profiler.end_section()
            
        else:
            input = (image_raw, pack_settings)
            
        profiler.save_to_profiling(self.profiling)
            
        if self.give_meta:
            return input, gt_image, {'id': in_filename[:8], 'ratio': pack_settings['exposure_ratio']}
        else:
            return input, gt_image