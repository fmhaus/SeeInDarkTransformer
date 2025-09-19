import torch
import torch.nn as nn
from util import image_util

class Model(torch.nn.Module):
    def __init__(self):
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
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.max_pool(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.max_pool(conv2)
         
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.max_pool(conv3)
    
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.max_pool(conv4)
           
        # connect zeros in up6
        up6 = torch.cat((torch.zeros(conv4.shape), conv4), dim = 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
          
        up7 = torch.cat((self.up7(conv6), conv3), dim = 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
          
        up8 = torch.cat((self.up8(conv7), conv2), dim = 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = torch.cat((self.up9(conv8), conv1), dim = 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        conv10 = self.conv10(conv9)
        
        return image_util.depth_to_space(conv10, 2)

    def load_state(self, path='./models/sony_images/states/sid_no_bottleneck.pt'):
        self.load_state_dict(torch.load(path, weights_only=True))