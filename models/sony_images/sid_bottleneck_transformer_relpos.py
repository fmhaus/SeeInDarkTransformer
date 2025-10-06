import torch
import torch.nn as nn
import torch.nn.functional as F
from util import image_util

# Implementation with direction aware relative positions
class RelPosBiases(nn.Module):
    def __init__(self, n_heads, head_dim, max_window_size):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        
        self.max_height, self.max_width = max_window_size

        self.bias_table = nn.Parameter(torch.zeros((n_heads, 2*self.max_height - 1, 2*self.max_width - 1), dtype=torch.float32))

    def forward(self, shape):
        h, w = shape
        assert h <= self.max_height
        assert w <= self.max_width

        y_coords = torch.arange(h)
        x_coords = torch.arange(w)

        coords = torch.stack(torch.meshgrid((y_coords, x_coords), indexing='ij'))
        coords_flatten = coords.flatten(1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # move range from (-max_height, max_height) to [0, 2*max_height - 1)
        relative_coords[0] += self.max_height - 1
        relative_coords[1] += self.max_width - 1

        return self.bias_table[:, relative_coords[0], relative_coords[1]].unsqueeze(0)

# Implementation with symmetric relative positions (same bias for a pixel to the left as to the right)
class RelPosBiasesSym(nn.Module):
    def __init__(self, n_heads, head_dim, max_window_size):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        
        self.max_height, self.max_width = max_window_size

        self.bias_table = nn.Parameter(torch.zeros((n_heads, self.max_height, self.max_width), dtype=torch.float32))

    
    def forward(self, shape):
        h, w = shape
        assert h <= self.max_height
        assert w <= self.max_width

        y_coords = torch.arange(h)
        x_coords = torch.arange(w)

        coords = torch.stack(torch.meshgrid((y_coords, x_coords), indexing='ij'))
        coords_flatten = coords.flatten(1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.abs() # mirror symmetry

        return self.bias_table[:, relative_coords[0], relative_coords[1]].unsqueeze(0)


class MultiheadSelfAttentionRelPos(nn.Module):
    def __init__(self, dim_model, n_heads, max_window_size, dropout = 0.0, relpos_type = 'direction_aware'):
        super().__init__()
        self.dim_model = dim_model
        self.n_heads = n_heads

        self.head_dim = dim_model // n_heads
        assert self.head_dim * n_heads == dim_model

        if relpos_type == 'direction_aware' or relpos_type is None:
            self.rel_pos_biases = RelPosBiases(n_heads, self.head_dim, max_window_size)
        elif relpos_type == 'symmetric':
            self.rel_pos_biases = RelPosBiasesSym(n_heads, self.head_dim, max_window_size)
        else:
            raise ValueError('Invalid relpos_type')

        self.qkv_proj = nn.Linear(dim_model, dim_model * 3)
        self.out_proj = nn.Linear(dim_model, dim_model)

        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)

        self.scaling = self.head_dim ** -0.5
    
    def forward(self, x):
        B, H, W, D = x.shape
        assert D == self.dim_model
        S = H * W  # sequence length

        # fuse H and W
        x = x.reshape(B, S, D)

        # do projection all at once
        qkv = self.qkv_proj(x)
        # to [3 (qkv), batch_size, n_heads, seq_len, head_dim]
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # output is [batch_size, n_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # apply pos biases
        attn_scores = attn_scores + self.rel_pos_biases([H, W])

        # scale softmax for numerical stability
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values
        attn_weights = self.softmax(attn_scores)

        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v) # [batch_size, n_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B, S, D) # [batch_size, seq_len, dim_model]
        
        out = self.out_proj(attn_output)
        return out.reshape(B, H, W, D)

class TransformerBlock(nn.Module):
    def __init__(self, dim_model, n_heads, max_window_size, dim_ff=None, attn_dropout = 0.0, mlp_dropout = 0.0, relpos_type = None):
        super().__init__()
        
        dim_ff = dim_ff or 4 * dim_model
        
        self.attention = MultiheadSelfAttentionRelPos(dim_model, n_heads, max_window_size, dropout=attn_dropout, relpos_type=relpos_type)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(dim_ff, dim_model)
        )
        
        self.norm1 = nn.LayerNorm(dim_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(attn_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)
        
        self.apply(self._init_weights)
    
    def forward(self, x):
        # use pre layer norm
        attention_output = self.attention(self.norm1(x))
        x = x + self.dropout1(attention_output)
        
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout2(mlp_output)
        
        return x
    
    # Truncation normal initialization
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class TransformerBottleneck(nn.Module):
    def __init__(self, n_transformer_blocks, in_channels, dim_model, relpos_type=None):
        super().__init__()
        
        self.down = nn.Conv2d(in_channels, dim_model, 2, stride=2)

        max_window_size = (45, 67)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim_model, 4, max_window_size, relpos_type=relpos_type, attn_dropout=0.1, mlp_dropout=0.2) 
                for _ in range(n_transformer_blocks)])
        
        self.up = nn.ConvTranspose2d(dim_model, in_channels, 2, stride=2)
    
    def forward(self, x):
        # need padding for odd dimension
        window_size = x.shape[2:]
        down = self.down(self.pad_to_even(x))
        # [N, 256, 45, 67]

        embeddings = down.permute(0, 2, 3, 1) # to [N, H, W, C]
        
        for block in self.blocks:
            embeddings = block(embeddings)
        
        nchw_embeddings = embeddings.permute(0, 3, 1, 2) # to [N, C, H, W]

        up = self.up(nchw_embeddings)
        return self.crop_like(up, window_size)
    
    def set_dropout(self, attn_dropout, mlp_dropout):
        for block in self.blocks:
            block.set_dropout(attn_dropout, mlp_dropout)
    
    def pad_to_even(self, x):
        _, _, H, W = x.shape
        pad_h = H & 1
        pad_w = W & 1
        
        return F.pad(x, (0, pad_w, 0, pad_h), "replicate")
    
    def crop_like(self, x, size):
       return x[..., :size[0], :size[1]]

class Model(nn.Module):
    def __init__(self, transformer_blocks, relpos_type=None):
        super().__init__()
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool2d(2)
        
        self.conv1_1 = nn.Conv2d(4, 32, 3, padding='same')
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding='same')
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding='same')
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding='same')
        
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding='same')
        
        self.bottleneck5 = TransformerBottleneck(transformer_blocks, 256, 256, relpos_type=relpos_type)
                
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding='same')
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding='same')
        
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding='same')
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding='same')
        
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding='same')
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding='same')
        
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding='same')
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding='same')
        
        self.conv10 = nn.Conv2d(32, 12, 1, padding='same')
    
    def forward(self, x):
        _, C, H, W = x.shape
        assert C == 4 and H == 1424 and W == 2128
        # [N, 4, 1424, 2128]
        
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.max_pool(conv1)
        # [N, 32, 712, 1064]  
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.max_pool(conv2)
        # [N, 64, 356, 532]
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.max_pool(conv3)
        # [N, 128, 178, 266]
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.max_pool(conv4)
        # [N, 256, 89, 133]
        
        bottleneck5 = self.bottleneck5(pool4)
        
        out5 = torch.cat((bottleneck5, pool4), dim=1)
        # [N, 256, 89, 133]
           
        up6 = torch.cat((self.up6(out5), conv4), dim = 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        # [N, 256, 178, 266]
          
        up7 = torch.cat((self.up7(conv6), conv3), dim = 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        # [N, 128, 356, 532]
          
        up8 = torch.cat((self.up8(conv7), conv2), dim = 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        # [N, 64, 712, 1064]
    
        up9 = torch.cat((self.up9(conv8), conv1), dim = 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        # [N, 32, 1424, 2128]
        
        conv10 = self.conv10(conv9)
        # [N, 12, 1424, 2128]
        
        # depth_to_space in pytorch
        return image_util.depth_to_space(conv10, 2)
    
    def set_transformer_dropout(self, attn_dropout, mlp_dropout):
        self.bottleneck5.set_dropout(attn_dropout, mlp_dropout)
    
    def load_state(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

class Model_2b(Model):
    def __init__(self):
        super().__init__(2, relpos_type='direction_aware')
    
    def load_pretrained(self):
        raise RuntimeError('No pretrained model')
    
class Model_2b_Sym(Model):
    def __init__(self):
        super().__init__(2, relpos_type='symmetric')
    
    def load_pretrained(self):
        raise RuntimeError('No pretrained model')