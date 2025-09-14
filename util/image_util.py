import random
import torch
import torch.nn.functional as F
import math

def augment_mirror(item):
    in_image, gt_image = item
    
    if random.choice([True, False]):
        in_image = torch.flip(in_image, [1])
        gt_image = torch.flip(gt_image, [1])
    if random.choice([True, False]):
        in_image = torch.flip(in_image, [2])
        gt_image = torch.flip(gt_image, [2])
    return (in_image, gt_image)

def augment_crop_rescale(item, min_scale=0.5):
    in_image, gt_image = item
    
    _, H, W = in_image.shape
    
    width = W * random.uniform(min_scale, 1.0)
    height = width * H / W
    
    width = int(width)
    height = int(height)
    
    x1 = random.randint(0, W-width)
    y1 = random.randint(0, H-height)
    
    in_crop = in_image[:, y1:y1+height, x1:x1+width]
    gt_crop = gt_image[:, 2*y1:2*(y1+height), 2*x1:2*(x1+width)]
    
    in_resized = F.interpolate(in_crop.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
    gt_resized = F.interpolate(gt_crop.unsqueeze(0), size=(2*H, 2*W), mode='bicubic', align_corners=False).squeeze(0)
    
    return in_resized.clip(0.0, 1.0), gt_resized.clip(0.0, 1.0)

class AugmentCropRescale():
    def __init__(self, min_scale=0.5):
        self.min_scale = min_scale
    
    def __call__(self, *args, **kwargs):
        return augment_crop_rescale(args[0], self.min_scale)

def augment_translate_pad(item, max_translate_factor=0.2): # mode reflect or replicate
    in_image, gt_image = item
    _, H, W = in_image.shape
    
    trans_x = random.randint(-int(max_translate_factor * W), int(max_translate_factor * W))
    trans_y = random.randint(-int(max_translate_factor * H), int(max_translate_factor * H))
    
    in_image = torch.roll(in_image, shifts=(trans_y, trans_x), dims=(1, 2))
    gt_image = torch.roll(gt_image, shifts=(2*trans_y, 2*trans_x), dims=(1, 2))
    
    pad_left = trans_x if trans_x > 0 else 0
    pad_right = -trans_x if trans_x < 0 else 0
    pad_top = trans_y if trans_y > 0 else 0
    pad_bottom = -trans_y if trans_y < 0 else 0
    
    in_image = F.pad(in_image[:, pad_top:(H-pad_bottom), pad_left:(W-pad_right)], pad=[pad_left, pad_right, pad_top, pad_bottom], mode='reflect')
    gt_image = F.pad(gt_image[:, 2*pad_top:2*(H-pad_bottom), 2*pad_left:2*(W-pad_right)], pad=[2*pad_left, 2*pad_right, 2*pad_top, 2*pad_bottom], mode='reflect')
    
    return (in_image, gt_image)

class AugmentTranslatePad():
    def __init__(self, max_translate_factor = 0.2):
        self.max_translate_factor = max_translate_factor
    
    def __call__(self, *args, **kwargs):
        return augment_translate_pad(args[0], self.max_translate_factor)

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
    mse = torch.clamp(mse, min=1e-10)
    
    psnr = 20.0 * math.log10(max_val) -10.0 * torch.log10(mse)
    return psnr

def tensor_to_images(tensor, flip_to_bgr = False):
    tensor = (tensor.clip(0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    tensor = torch.permute(tensor, (0, 2, 3, 1))
    if flip_to_bgr:
        tensor = torch.flip(tensor, [3])
    return tensor.numpy(force=True)

def depth_to_space(tensor, R):
    N, C_in, H, W = tensor.shape
    C_out = C_in // (R*R)
    depth = tensor.view(N, R, R, C_out, H, W)
    space = depth.permute(0, 3, 4, 1, 5, 2).reshape(N, C_out, H*2, W*2)
    return space