from models.sony_images import sid_bottleneck_transformer, dataset
from util import file_storage, time_util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import rawpy
import os
import io
import json

BATCH_SIZE = 1              # Amount of images processed simultaneously
EFFECTIVE_BATCH_SIZE = 32   # Amount of images before updating gradients
LEARNING_RATE = 1e-4
NUM_EPOCHS = 4001
LOG_INTERVAL = 1
NUM_WORKERS = 4

assert EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0

DATASET_FOLDER = 'E:/workspace/Bach/Bach300/Learning-to-See-in-the-Dark/dataset/'
PREPROCESSED_GT_FOLDER = 'E:/workspace/Bach/Bach300/BottleneckTransformerTrain/v1/preprocessed_gts/'
S3_OUT_FOLDER = 'train_out/test1/'

class RawImageDataset(torch.utils.data.Dataset):
    def __init__(self, list, augment_dataset):
        self.items = list
        self.augment_dataset = augment_dataset
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        in_name = self.items[idx][0]
        in_file = os.path.join(DATASET_FOLDER, in_name[2:])
        in_exposure = float(os.path.basename(in_name)[9:-5])
        
        gt_name = self.items[idx][1]
        gt_filename = os.path.basename(gt_name)
        gt_exposure = float(gt_filename[9:-5])
        
        raw = rawpy.imread(in_file)
        in_image = sid_bottleneck_transformer.get_model_input(raw, gt_exposure / in_exposure)
        
        gt_preprocessed = os.path.join(PREPROCESSED_GT_FOLDER, gt_filename + '.npy')
        gt_image = torch.from_numpy(np.load(gt_preprocessed, allow_pickle=True))
        
        if self.augment_dataset:
            return dataset.augment_training_data(in_image, gt_image)
        else:
            return in_image, gt_image

def preprocess_ground_truths(list):
    
    os.makedirs(PREPROCESSED_GT_FOLDER, exist_ok=True)
    
    # get first entry to list to find path
    gt_name = list[0][1]
    gt_folder = os.path.dirname(os.path.join(DATASET_FOLDER, gt_name[2:]))
    
    for filename in os.listdir(gt_folder):
        npy_file = os.path.join(PREPROCESSED_GT_FOLDER, filename + '.npy')
        if not os.path.exists(npy_file):
            raw_file = os.path.join(gt_folder, filename)
            gt = rawpy.imread(raw_file).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            tensor = np.permute_dims(np.float32(gt / 65535.0), axes=(2, 0, 1))
            np.save(npy_file, tensor)
            
            print(f"Processed: {npy_file}")

def get_list(object_storage, key):
    lines = object_storage.load_file(key).splitlines()
    return list(map(lambda x: x.split(' '), lines))

def get_psnr(out, gt):
    mse = ((out - gt)**2).mean(dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-10)
    
    psnr = -10.0 * torch.log10(mse)
    return psnr.mean().item()

if __name__ == '__main__':
    with open('s3_access.json', 'r') as file:
        userdata = json.load(file)

    object_storage = file_storage.S3ObjectStorage(userdata)

    # set up training and validation data
    train_list = get_list(object_storage, 'datasets/SID/Sony_train_list.txt')
    val_list = get_list(object_storage, 'datasets/SID/Sony_val_list.txt')

    preprocess_ground_truths(train_list)
    print('Preprocessed Ground-Truths.')

    dataset = RawImageDataset(train_list, augment_dataset=False)
    dataset_val = RawImageDataset(val_list, augment_dataset=False)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=NUM_WORKERS > 0
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=NUM_WORKERS > 0
    )

    # load model
    model = sid_bottleneck_transformer.Model()
    model.load_pretrained('./models/sony_images/states/bottleneck_transformer_initial.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    gradient_accumulation_steps = EFFECTIVE_BATCH_SIZE // BATCH_SIZE

    time_estimator = time_util.TimeEstimator()

    for epoch in range(0, NUM_EPOCHS):
        
        # to training step
        time_estimator.reset()
        total_loss = 0.0
        
        model.train()
        optimizer.zero_grad()
        
        for batch_idx, (in_images, gt_images) in enumerate(dataloader):
            in_images = in_images.to(device, non_blocking=True)
            gt_images = gt_images.to(device, non_blocking=True)
            
            out_images = model(in_images)
            
            loss = criterion(out_images, gt_images)
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps * in_images.shape[0]
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            time_estimator.next_interval()
            
            if batch_idx % LOG_INTERVAL == 0:
                print(f'Training Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}/{math.ceil(len(dataset) / BATCH_SIZE)}], Loss: {loss.item():.4f} ' +
                      f'TimePassed: {time_util.format_elapsed_time(time_estimator.time_passed_total())} ' + 
                      f'TimeLeftEstimate: {time_util.format_elapsed_time(time_estimator.get_ema() * (len(dataset) - (batch_idx+1) * BATCH_SIZE))}')
        
        avg_train_loss = total_loss / len(dataset)
        train_time = time_estimator.time_passed_total()
        
        model.eval()
        
        total_loss = 0
        total_psnr = 0
        time_estimator.reset()
        
        with torch.no_grad():
            for batch_idx, (in_images, gt_images) in enumerate(dataloader_val):
                batch_size = in_images.shape[0]
                
                in_images = in_images.to(device, non_blocking=True)
                gt_images = gt_images.to(device, non_blocking=True)
                
                out_images = model(in_images)
                out_images = out_images.clip(0.0, 1.0)
                
                loss = criterion(out_images, gt_images)
                total_loss += loss.item() * batch_size
                
                total_psnr += get_psnr(out_images, gt_images) * batch_size
                
                time_estimator.next_interval()
            
                if batch_idx % LOG_INTERVAL == 0:
                    print(f'Validation Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}/{math.ceil(len(dataset_val) / BATCH_SIZE)}], Loss: {loss.item():.4f} ' +
                        f'TimePassed: {time_util.format_elapsed_time(time_estimator.time_passed_total())} ' + 
                        f'TimeLeftEstimate: {time_util.format_elapsed_time(time_estimator.get_ema() * (len(dataset_val) - (batch_idx+1) * BATCH_SIZE))}')
        
        val_time = time_estimator.time_passed_total()
        avg_val_loss = total_loss / len(dataset_val)
        avg_psnr = total_psnr / len(dataset_val)
        
        stats = {
            'epoch': epoch,
            'batch_size': BATCH_SIZE,
            'effective_batch_size': EFFECTIVE_BATCH_SIZE,
            'model': 'sid_bottleneck_transformer_v1',
            'train_time': train_time,
            'val_time': val_time,
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
            'avg_val_psnr': avg_psnr,
            'learning_rate': LEARNING_RATE,
            'model_state_dict': S3_OUT_FOLDER + f'model_state_{epoch+1}.pt',
            'optimizer_state_dict': S3_OUT_FOLDER + f'optimizer_state_{epoch+1}.pt'
        }
        
        object_storage.store_file(S3_OUT_FOLDER + f'log_{epoch+1}.json', json.dumps(stats), binary=False)
        
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        object_storage.store_file(stats['model_state_dict'], buffer, binary=True)
        
        buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), buffer)
        buffer.seek(0)
        object_storage.store_file(stats['optimizer_state_dict'], buffer, binary=True)
        
        print(f'Saved epoch {epoch+1}. train loss: {avg_train_loss}, val loss: {avg_val_loss}')