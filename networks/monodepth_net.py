from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)
        
    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace = True)
    

class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
    
    
class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size
        
    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)
    

class resconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride) 
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)
        
    def forward(self, x):
#         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)

    
class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)
        
    def forward(self, x):
#         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)

    
def resblock(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks-1):
        layers.append(resconv(4*num_out_layers, num_out_layers, 1))
    layers.append(resconv(4*num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)
       
    
class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.up_nn = nn.Upsample(scale_factor=scale)
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        
    def forward(self, x):
        x = self.up_nn(x)
        return self.conv1(x)
    
    
class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        
    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return 0.3*F.sigmoid(x)
    
    
class resnet50_md(nn.Module):
    def __init__(self, num_in_layers):
        super(resnet50_md, self).__init__()
        #encoder
        self.conv1 = conv(num_in_layers, 64, 7, 2) # H/2  -   64D
        self.pool1 = maxpool(3) # H/4  -   64D
        self.conv2 = resblock(64, 64, 3, 2) # H/8  -  256D
        self.conv3 = resblock(256, 128, 4, 2) # H/16 -  512D
        self.conv4 = resblock(512, 256, 6, 2) # H/32 - 1024D
        self.conv5 = resblock(1024, 512, 3, 2) # H/64 - 2048D
        
        #decoder
        self.upconv6 = upconv(2048, 512, 3, 2)
        self.iconv6 = conv(1024+512, 512, 3, 1)
        
        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(512+256, 256, 3, 1)
        
        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(256+128, 128, 3, 1)
        self.disp4_layer = get_disp(128)
        self.udisp4_layer = nn.Upsample(scale_factor=2)
        
        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64+2, 64, 3, 1)
        self.disp3_layer = get_disp(64)
        self.udisp3_layer = nn.Upsample(scale_factor=2)
        
        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(32+64+2, 32, 3, 1)
        self.disp2_layer = get_disp(32)
        self.udisp2_layer = nn.Upsample(scale_factor=2)
        
        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        #encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        #skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4
        
        #decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)
        
        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)
        
        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = self.udisp4_layer(self.disp4)
        
        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = self.udisp3_layer(self.disp3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = self.udisp2_layer(self.disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4
    
class resnet50_decoder(nn.Module):
    def __init__(self):
        super(resnet50_decoder, self).__init__()
        
        # self.skip_layers = [conv1_l, conv2_l, in_conv3b, conv3b, conv4b, conv5b]
        #decoder
        self.upconv6 = upconv(1024, 512, 3, 2)
        self.iconv6 = conv(512+512, 512, 3, 1)
        
        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(512+256, 256, 3, 1)
        
        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(256+128, 128, 3, 1)
        self.disp4_layer = get_disp(128)
        self.udisp4_layer = nn.Upsample(scale_factor=2)
        
        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(128+64+2, 64, 3, 1)
        self.disp3_layer = get_disp(64)
        self.udisp3_layer = nn.Upsample(scale_factor=2)
        
        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64+32+2, 32, 3, 1)
        self.disp2_layer = get_disp(32)
        self.udisp2_layer = nn.Upsample(scale_factor=2)
        
        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):

        # self.skip_layers = [conv1_l, conv2_l, in_conv3b, conv3b, conv4b, conv5b]
        #skips
        skip1 = x[0]
        skip2 = x[1]
        skip3 = x[2]
        skip4 = x[3]
        skip5 = x[4]
        x5 = x[5]
        
        #decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)
        
        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)
        
        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = self.udisp4_layer(self.disp4)
        
        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = self.udisp3_layer(self.disp3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = self.udisp2_layer(self.disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4
    
    def weight_parameters(self):
	return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
	return [param for name, param in self.named_parameters() if 'bias' in name]

    
class resnet18_md(nn.Module):
    def __init__(self, num_in_layers):
        super(resnet18_md, self).__init__()
        #encoder
        self.conv1 = conv(num_in_layers, 64, 7, 2) # H/2  -   64D
        self.pool1 = maxpool(3) # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2) # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2) # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2) # H/32 - 256D
        self.conv5 = resblock_basic(256, 512, 2, 2) # H/64 - 512D
        
        #decoder
        self.upconv6 = upconv(512, 512, 3, 2)
        self.iconv6 = conv(256+512, 512, 3, 1)
        
        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(128+256, 256, 3, 1)
        
        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(64+128, 128, 3, 1)
        self.disp4_layer = get_disp(128)
        self.udisp4_layer = nn.Upsample(scale_factor=2)
        
        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64, 64, 3, 1)
        self.disp3_layer = get_disp(64)
        self.udisp3_layer = nn.Upsample(scale_factor=2)
        
        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64+32, 32, 3, 1)
        self.disp2_layer = get_disp(32)
        self.udisp2_layer = nn.Upsample(scale_factor=2)
        
        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        #encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        #skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4
        
        #decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)
        
        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)
        
        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = self.udisp4_layer(self.disp4)
        
        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = self.udisp3_layer(self.disp3)
        
        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = self.udisp2_layer(self.disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4
