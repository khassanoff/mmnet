import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
import pdb


class ImageEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 2, 2), (1, 1, 1))
        self.conv4 = nn.Conv3d(96, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1))

        #Recurrecnt layers
        self.gru1  = nn.GRU(128*6*8, 256, 1, bidirectional=True) # images
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)

        #Fully connected layers
        self.FC1   = nn.Linear(512, 64) 
        self.FC2   = nn.Linear(64*opt.num_frames, 64)

        #Activation functions
        self.relu = nn.ReLU(inplace=True)
    
        #Dropout
        self.dropout = nn.Dropout(opt.drop)
        self.dropout3d = nn.Dropout3d(opt.drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
 
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
 
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)
 
        x = self.FC1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class AudioEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(AudioEncoder, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 2, 2), (1, 1, 1))
        self.conv4 = nn.Conv3d(96, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1))

        #Recurrecnt layers
        self.gru1  = nn.GRU(128*8, 256, 1, bidirectional=True) # audio
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)

        #Fully connected layers
        self.FC1   = nn.Linear(512, 64) 
        self.FC2   = nn.Linear(64*(int(opt.segment_len*opt.sample_rate)//512+1), 64)    # audio

        #Activation functions
        self.relu = nn.ReLU(inplace=True)
    
        #Dropout
        self.dropout = nn.Dropout(opt.drop)
        self.dropout3d = nn.Dropout3d(opt.drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
 
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
 
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)
 
        x = self.FC1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SFNet(torch.nn.Module):
    def __init__(self, opt):
        super(SFNet, self).__init__()
        self.opt = opt

        if opt.mode in [1,4,7]:
            self.rgb_enc = ImageEncoder(self.opt)
        if opt.mode in [2,4,7]:
            self.thr_enc = ImageEncoder(self.opt)
        if opt.mode in [3,7]:
            self.aud_enc = AudioEncoder(self.opt)


        if opt.mode in [1,2,3]:
            self.FC1 = nn.Linear(64, 1)
        elif opt.mode in [4,5,6]:
            self.FC1 = nn.Linear(64*2, 1)
        elif opt.mode in [7]:
            self.FC1 = nn.Linear(64*3, 1)

    def forward(self, x1=None, x2=None, x3=None):
        if self.opt.mode in [1,4,7]:
            x1 = self.rgb_enc(x1)
        if self.opt.mode in [2,4,7]:
            x2 = self.thr_enc(x2)
        if self.opt.mode in [3,7]:
            x3 = self.aud_enc(x3)

        if self.opt.mode == 1:
            x  = self.FC1(x1)
        elif self.opt.mode == 2:
            x  = self.FC1(x2)
        elif self.opt.mode == 3:
            x  = self.FC1(x3)
        elif self.opt.mode == 4:
            x  = self.FC1(torch.cat((x1,x2),1))
        elif self.opt.mode == 7:
            x  = self.FC1(torch.cat((x1,x2,x3),1))
        return x

class LipNet(torch.nn.Module):
    def __init__(self, opt):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        #self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
 
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        #self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
 
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 2, 2), (1, 1, 1))
        #self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv4 = nn.Conv3d(96, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1))

        #self.gru1  = nn.GRU(128*6*8, 256, 1, bidirectional=True) # images
        self.gru1  = nn.GRU(128*8, 256, 1, bidirectional=True) # audio
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC1   = nn.Linear(512, 64) 
        #self.FC2   = nn.Linear(64*opt.num_frames, 1)   # images
        self.FC2   = nn.Linear(64*(int(opt.segment_len*opt.sample_rate)//512+1), 1)    # audio

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(opt.drop)
        self.dropout3d = nn.Dropout3d(opt.drop)
        #self._init()
 
    def _init(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)
 
        init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        init.constant_(self.conv4.bias, 0)

        init.kaiming_normal_(self.FC1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.FC2.weight, nonlinearity='sigmoid')
        init.constant_(self.FC1.bias, 0)
        init.constant_(self.FC2.bias, 0)
 
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        #x = self.pool1(x)
 
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        #x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        #x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout3d(x)
 
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
 
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
 
        x, h = self.gru1(x)
        x = self.dropout(x)
        x, h = self.gru2(x)
        x = self.dropout(x)
 
        x = self.FC1(x)
        x = self.relu(x)
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.FC2(x)
        return x
