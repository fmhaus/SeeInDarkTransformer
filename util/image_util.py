import random
import torch
import torch.nn.functional as F
import math

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
    
def augment_translate_new(item, max_translate_factor=0.25): # mode reflect or replicate
    in_image, gt_image = item
    N, C, H, W = in_image.shape
    device = in_image.device
    
    # max translation in pixels
    max_tx = int(max_translate_factor * W)  
    max_ty = int(max_translate_factor * H)
    
    # randomize translation for every batch
    tx = torch.randint(1-max_tx, max_tx, [N], device=device)
    ty = torch.randint(1-max_ty, max_ty, [N], device=device)
    
    def reflect_indices(translation, end, dim):
        # get indices in [N, H] shape (or W) and apply translation
        indices = torch.arange(end, device=device)[None, :] - translation[:, None]
        # Indices outside of (0:N) are reflected using abs
        m = end-1
        reflect = torch.abs((indices + m) % (2*m) - m)
        # expand to 4 dims
        expanded = reflect[:, :, None, None].movedim(1, dim)
        return expanded
    
    batch_indices = torch.arange(N, device=device)[:, None, None, None]
    in_channel_indices = torch.arange(C, device=device)[None, :, None, None]
    gt_channel_indices = torch.arange(gt_image.shape[1], device=device)[None, :, None, None]
    in_image = in_image[batch_indices, in_channel_indices, reflect_indices(ty, H, 2), reflect_indices(tx, W, 3)]
    gt_image = gt_image[batch_indices, gt_channel_indices, reflect_indices(2*ty, 2*H, 2), reflect_indices(2*tx, 2*W, 3)]

    return (in_image, gt_image)

def augment_translate_old_loop(item, max_translate_factor=0.25):
    in_image, gt_image = item
    
    N, _, H, W = in_image.shape
    
    for i in range(N):
    
        trans_x = random.randint(-int(max_translate_factor * W), int(max_translate_factor * W))
        trans_y = random.randint(-int(max_translate_factor * H), int(max_translate_factor * H))
        
        in_image[i] = torch.roll(in_image[i], shifts=(trans_y, trans_x), dims=(1, 2))
        gt_image[i] = torch.roll(gt_image[i], shifts=(2*trans_y, 2*trans_x), dims=(1, 2))
        
        pad_left = trans_x if trans_x > 0 else 0
        pad_right = -trans_x if trans_x < 0 else 0
        pad_top = trans_y if trans_y > 0 else 0
        pad_bottom = -trans_y if trans_y < 0 else 0
        
        in_image[i] = F.pad(in_image[i, :, pad_top:(H-pad_bottom), pad_left:(W-pad_right)], pad=[pad_left, pad_right, pad_top, pad_bottom], mode='reflect')
        gt_image[i] = F.pad(gt_image[i, :, 2*pad_top:2*(H-pad_bottom), 2*pad_left:2*(W-pad_right)], pad=[2*pad_left, 2*pad_right, 2*pad_top, 2*pad_bottom], mode='reflect')
    
    return in_image, gt_image

def augment_translate_old(item, max_translate_factor=0.25):
    in_image, gt_image = item
    
    H, W = in_image.shape[-2:]
    
    trans_x = random.randint(-int(max_translate_factor * W), int(max_translate_factor * W))
    trans_y = random.randint(-int(max_translate_factor * H), int(max_translate_factor * H))
    
    in_image = torch.roll(in_image, shifts=(trans_y, trans_x), dims=(1, 2))
    gt_image = torch.roll(gt_image, shifts=(2*trans_y, 2*trans_x), dims=(1, 2))
    
    pad_left = trans_x if trans_x > 0 else 0
    pad_right = -trans_x if trans_x < 0 else 0
    pad_top = trans_y if trans_y > 0 else 0
    pad_bottom = -trans_y if trans_y < 0 else 0
    
    in_image = F.pad(in_image[:, :, pad_top:(H-pad_bottom), pad_left:(W-pad_right)], pad=[pad_left, pad_right, pad_top, pad_bottom], mode='reflect')
    gt_image = F.pad(gt_image[:, :, 2*pad_top:2*(H-pad_bottom), 2*pad_left:2*(W-pad_right)], pad=[2*pad_left, 2*pad_right, 2*pad_top, 2*pad_bottom], mode='reflect')
    
    return in_image, gt_image

def augment_translate_fast(item, max_translate_factor=0.25):
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
            return augment_translate_fast(args[0], self.max_translate_factor)

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