import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util import image_util

class TransformerBlock(nn.Module):
    def __init__(self, dim_model, n_heads, dim_ff=None, dropout_attn=0.1, dropout_mlp=0.1):
        super(TransformerBlock, self).__init__()
        
        dim_ff = dim_ff or 4 * dim_model
        
        self.attention = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout_attn, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout_mlp),
            nn.Linear(dim_ff, dim_model)
        )
        
        self.norm1 = nn.LayerNorm(dim_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout_attn)
        self.dropout2 = nn.Dropout(dropout_mlp)
        
        self.apply(self._init_weights)
    
    # Truncation normal initialization, overkill probably
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    
    def forward(self, x):
        # use pre layer norm
        q = k = v = self.norm1(x)
        attention_output, _ = self.attention(q, k, v)
        x = x + self.dropout1(attention_output)
        
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout2(mlp_output)
        
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
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
        
        self.down5 = nn.Conv2d(256, 256, 2, stride=2)
        
        self.register_buffer('pos_embed5', self.create_positional_embeddings_2D((45, 67), 256), persistent=False)
        
        self.transformer5_1 = TransformerBlock(256, 4)
        self.transformer5_2 = TransformerBlock(256, 4)
        self.transformer5_3 = TransformerBlock(256, 4)
        #self.transformer5_4 = TransformerBlock(256, 4)
        
        self.up5 = nn.ConvTranspose2d(256, 256, 2, stride=2)
                
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
        _N, C, H, W = x.shape
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
        
        # need padding for odd dimension
        _, _, h5, w5 = pool4.shape
        down5 = self.down5(self.pad_to_even(pool4))
        down5_shape = down5.shape
        
        # transform to transformer shape
        embed5 = down5.flatten(2).transpose(1, 2)
        embed5 = embed5 + self.pos_embed5
        # [N, 45*67, 256]
        
        embed5 = self.transformer5_1(embed5)
        embed5 = self.transformer5_2(embed5)
        embed5 = self.transformer5_3(embed5)
        #embed5 = self.transformer5_4(embed5)
        
        # return to conv shape
        up5 = self.up5(embed5.transpose(1, 2).reshape(down5_shape))
        up5 = self.crop_like(up5, (h5, w5))
        
        # adding a skip connection around the transformer bottleneck
        # Im unsure about this one because it can cause the model to just ignore transformer features 
        # and only rely on the skip connection
        out5 = torch.cat((up5, pool4), dim=1)
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

    def pad_to_even(self, x):
        _N, _C, H, W = x.shape
        pad_h = H & 1
        pad_w = W & 1
        
        return F.pad(x, (0, pad_w, 0, pad_h), "replicate")
    
    def crop_like(self, x, size):
       _N, _C, H, W = x.shape
       return x[..., :size[0], :size[1]]

    def create_positional_embeddings_1D(self, window_size, dim_model):
        pos = torch.arange(window_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float) * (-math.log(10000.0) / dim_model))
        
        pos_embeds = torch.empty(window_size, dim_model)
        pos_embeds[:, 0::2] = torch.sin(pos * div_term)
        pos_embeds[:, 1::2] = torch.cos(pos * div_term)
        
        return pos_embeds
    
    def create_positional_embeddings_2D(self, dimensions, dim_model):
        h, w = dimensions
        d = dim_model // 2
        
        row_embeds = self.create_positional_embeddings_1D(h, d).unsqueeze(1)
        col_embeds = self.create_positional_embeddings_1D(w, d).unsqueeze(0)
        
        return torch.cat((row_embeds.expand(h, w, d), col_embeds.expand(h, w, d)), dim=2).view(h*w, dim_model)
        
    
    def load_pretrained(self, file):
        self.load_state_dict(torch.load(file, weights_only=True))