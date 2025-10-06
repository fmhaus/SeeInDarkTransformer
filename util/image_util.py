import random
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2

try:
    from google.colab.patches import cv2_imshow
    google_colab = True
except:
    google_colab = False

def augment_mirror(item):
    in_image, gt_image = item
    
    # In an ideal world, every image in the batch would have its own flip chance
    if random.choice([True, False]):
        in_image = torch.flip(in_image, [-2])
        gt_image = torch.flip(gt_image, [-2])
    if random.choice([True, False]):
        in_image = torch.flip(in_image, [-1])
        gt_image = torch.flip(gt_image, [-1])
    return (in_image, gt_image)

def augment_crop(item, size_step=2**5, min_size_factor=0.5):
    in_image, gt_image = item

    h, w = in_image.shape[2:]

    down_h = h // size_step
    down_w = w // size_step

    crop_h = torch.randint(int(min_size_factor * h), down_h) * size_step
    crop_w = torch.randint((int(min_size_factor * w)), down_w) * size_step

    x = torch.randint(0, h-crop_h)
    y = torch.randint(0, w-crop_w)

    in_image = in_image[:, :, y:y+crop_h, x:x+crop_w]
    gt_image = gt_image[:, :, 2*x:2*(y+crop_h), 2*x:2*(x+crop_w)]

    return (in_image, gt_image)

class AugmentCrop():
    def __init__(self, size_step=2**5, min_size_factor=0.5):
        self.size_step = size_step
        self.min_size_factor = min_size_factor
    
    def __call__(self, *args, **kwargs):
        return augment_crop(self.size_step, self.min_size_factor)

def augment_translate_reflect(item, max_translate_factor=0.25):
    in_image, gt_image = item
    
    H, W = in_image.shape[-2:]
    
    tx = random.randint(-int(max_translate_factor * W), int(max_translate_factor * W))
    ty = random.randint(-int(max_translate_factor * H), int(max_translate_factor * H))
    
    x1 = tx if tx > 0 else 0
    x2 = -tx if tx < 0 else 0
    y1 = ty if ty > 0 else 0
    y2 = -ty if ty < 0 else 0
    
    in_image = F.pad(in_image[:, :, y2:(H-y1), x2:(W-x1)], pad=[x1, x2, y1, y2], mode='reflect')
    gt_image = F.pad(gt_image[:, :, 2*y2:2*(H-y1), 2*x2:2*(W-x1)], pad=[2*x1, 2*x2, 2*y1, 2*y2], mode='reflect')
    
    return in_image, gt_image

class AugmentTranslateReflect():
    def __init__(self, max_translate_factor = 0.2, chance = 1):
        self.max_translate_factor = max_translate_factor
        self.chance = chance
    
    def __call__(self, *args, **kwargs):
        if random.uniform(0, 1) >= random.chance:
            return augment_translate_reflect(args[0], self.max_translate_factor)

class AugmentSequentiel():
    def __init__(self, *augments):
        self.augments = augments
    
    def __call__(self, *args, **kwargs):
        x = args[0]
        for augment in self.augments:
            x = augment(x)
        return x
        
def batch_psnr(img, gt, max_val = 1.0):
    mse = ((img - gt)**2).mean(dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-10)   # clamp for numeric stability
    psnr = 20.0 * math.log10(max_val) - 10.0 * torch.log10(mse)
    return psnr

def tensor_to_images(tensor):
    tensor = (tensor.clip(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    tensor = torch.permute(tensor, (0, 2, 3, 1))
    return tensor.numpy(force=True)

def images_flip_rgb_bgr(tensor):
    return np.flip(tensor, 3)

def depth_to_space(tensor, R):
    N, C_in, H, W = tensor.shape
    C_out = C_in // (R*R)
    depth = tensor.view(N, R, R, C_out, H, W)
    space = depth.permute(0, 3, 4, 1, 5, 2).reshape(N, C_out, H*2, W*2)
    return space

def show_image(title, image):
    if google_colab:
        if title:
            print(title)
        cv2_imshow(image)
    else:
        cv2.imshow(title, image)
        cv2.waitKey()
        cv2.destroyAllWindows()