import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from numbers import Number

def sample_diagonal_MultiGauss(mu, log_var, n):
    # reference :
    # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n

    # Convert z_log_var to std
    std = torch.exp(0.5 * log_var)
    def expand(v):
        if isinstance(v, Number):
            return torch.Tensor([v]).expand(n, 1)
        else:
            return v.expand(n, *v.size())
    if n != 1 :
        mu = expand(mu)
        std = expand(std)
    eps = Variable(std.data.new(std.size()).normal_().to(std.device))
    samples =  mu + eps * std
    samples = samples.reshape((n * mu.shape[1],)+ mu.shape[2:])
    return samples




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def poolOutDim(inDim, kernel_size, padding=0, stride=0, dilation=1):
    if stride == 0:
        stride = kernel_size
    num = inDim + 2*padding - dilation*(kernel_size - 1) - 1
    outDim = int(np.floor(num/stride + 1))
    return outDim

class ConvolutionalBackbone(nn.Module):
    def __init__(self, args):
        super(ConvolutionalBackbone, self).__init__()
        
        # basically using the number of dims and the number of poolings to be used 
        # figure out the size of the last fc layer so that this network is general to 
        # any images
        self.deterministic_encoder = args.deterministic_encoder
        self.args = args

        self.out_fc_dim = np.copy(args.img_dims)
        self.num_latent = args.emb_dims
        if (self.deterministic_encoder == True):
            self.z_dim = self.num_latent
            self.num_samples = 1
        else:
            self.z_dim = self.num_latent*2
            self.num_samples = args.num_samples

        padvals = [4, 8, 8]
        for i in range(3):
            self.out_fc_dim[0] = poolOutDim(self.out_fc_dim[0] - padvals[i], 2)
            self.out_fc_dim[1] = poolOutDim(self.out_fc_dim[1] - padvals[i], 2)
            self.out_fc_dim[2] = poolOutDim(self.out_fc_dim[2] - padvals[i], 2)
        
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, 12, 5)),
            ('bn1', nn.BatchNorm3d(12)),
            ('relu1', nn.PReLU()),
            ('mp1', nn.MaxPool3d(2)),

            ('conv2', nn.Conv3d(12, 24, 5)),
            ('bn2', nn.BatchNorm3d(24)),
            ('relu2', nn.PReLU()),
            ('conv3', nn.Conv3d(24, 48, 5)),
            ('bn3', nn.BatchNorm3d(48)),
            ('relu3', nn.PReLU()),
            ('mp2', nn.MaxPool3d(2)),

            ('conv4', nn.Conv3d(48, 96, 5)),
            ('bn4', nn.BatchNorm3d(96)),
            ('relu4', nn.PReLU()),
            ('conv5', nn.Conv3d(96, 192, 5)),
            ('bn5', nn.BatchNorm3d(192)),
            ('relu5', nn.PReLU()),
            ('mp3', nn.MaxPool3d(2)),
        ]))
        # input(self.out_fc_dim)
        self.fc = nn.Sequential(OrderedDict([
            ('flatten', Flatten()),
            ('fc1', nn.Linear(self.out_fc_dim[0]*self.out_fc_dim[1]*self.out_fc_dim[2]*192, 384)),
            ('relu6', nn.PReLU()),
            ('fc2', nn.Linear(384, 96)),
            ('relu7', nn.PReLU()),
            ('fc3', nn.Linear(96, self.z_dim))
        ]))

    def forward(self, x, num_samples=None):
        x = x.to(self.args.device)
        if(num_samples == None):
            num_samples = self.num_samples
        x_conv_features = self.conv(x)
        x_features = self.fc(x_conv_features)

        if(self.deterministic_encoder==True):
            
            zs = x_features
        else:
            if(num_samples == 1):
                
                zs = x_features
            else:
                z_mean = x_features[:, :self.num_latent]
                z_log_var = x_features[:, self.num_latent:]
                zs = sample_diagonal_MultiGauss(z_mean, z_log_var, num_samples)
            

        return zs
        


class ConvolutionalBackbone2D(nn.Module):
    def __init__(self, args):
        super(ConvolutionalBackbone2D, self).__init__()
        
        
        # # Define CNN layers for each slice
        # # Define batch normalization for each layer
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)),
            ('bn1', nn.BatchNorm2d(16)),
            ('relu1', nn.PReLU()),

            ('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            ('bn2', nn.BatchNorm2d(32)),
            ('relu2', nn.PReLU()),

            ('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.PReLU()),
            ]))
    

        
        # Adaptive pooling to ensure fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Adjust the size as needed
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Adjust size based on adaptive pooling
        self.relu4 = nn.PReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu5 = nn.PReLU()
        
    def forward_slice(self, x):
                
        x = self.conv(x)
        x = self.adaptive_pool(x)  # Adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        return x
    
    def forward(self, image_slice):
        # Encode each slice
        features = self.forward_slice(image_slice)

        return features

class OrthogonalEncoder(nn.Module):
    """docstring for OrthogonalEncoder"""
    def __init__(self, args):
        super(OrthogonalEncoder, self).__init__()
        self.args = args
        self.num_latent = args.emb_dims
        self.axial_encoder = ConvolutionalBackbone2D(args)
        self.sagittal_encoder = ConvolutionalBackbone2D(args)
        self.coronal_encoder = ConvolutionalBackbone2D(args)
        self.deterministic_encoder = args.deterministic_encoder
        if (self.deterministic_encoder == True):
            self.z_dim = self.num_latent
            self.num_samples = 1
        else:
            self.z_dim = self.num_latent*2
            self.num_samples = args.num_samples
        # self.fc2 = nn.Linear(1024 * 3, self.z_dim)  # Combine features from 3 slices
        self.fc = nn.Sequential(OrderedDict([
            ('fc_s1', nn.Linear(256*3, 256)),
            ('prelu1', nn.PReLU()), 
            ('fc_s2', nn.Linear(256, self.z_dim))
            ]))


    def forward(self, image,num_samples=None):
        if(num_samples == None):
            num_samples = self.num_samples
        # Encode each slice
        
        axial_feat = self.axial_encoder(image['axial'].to(self.args.device))
        coronal_feat = self.coronal_encoder(image['coronal'].to(self.args.device))
        sagittal_feat = self.sagittal_encoder(image['sagittal'].to(self.args.device))
        
        # Concatenate features from all three slices
        combined_feat = torch.cat((axial_feat, coronal_feat, sagittal_feat), dim=1)
        
        # Final fully connected layer
        features = self.fc(combined_feat)
        if(self.deterministic_encoder==True):
            
            zs = features
        else:
            if(num_samples == 1):
                
                zs = features
            else:
                z_mean = features[:, :self.num_latent]
                z_log_var = features[:, self.num_latent:]
                zs = sample_diagonal_MultiGauss(z_mean, z_log_var, num_samples)
            

        return zs





# Single slice encoder (only one view to be used for training and inference)

class SingleSliceEncoder(nn.Module):
    """docstring for SingleeSliceEncoder"""
    def __init__(self, args):
        super(SingleSliceEncoder, self).__init__()
        self.args = args
        self.num_latent = args.emb_dims
        
        self.encoder = ConvolutionalBackbone2D(args)

            
        self.deterministic_encoder = args.deterministic_encoder
        if (self.deterministic_encoder == True):
            self.z_dim = self.num_latent
            self.num_samples = 1
        else:
            self.z_dim = self.num_latent*2
            self.num_samples = args.num_samples
        # self.fc2 = nn.Linear(1024 * 3, self.z_dim)  # Combine features from 3 slices
        self.fc = nn.Sequential(OrderedDict([
            ('fc_s1', nn.Linear(256, 256)),
            ('prelu1', nn.PReLU()), 
            ('fc_s2', nn.Linear(256, self.z_dim))
            ]))


    def forward(self, image,num_samples=None):
        if(num_samples == None):
            num_samples = self.num_samples
        # Encode each slice
        slice_feature = self.encoder(image[self.args.use_slice].to(self.args.device))
        
        # Final fully connected layer
        features = self.fc(slice_feature)
        if(self.deterministic_encoder==True):
            
            zs = features
        else:
            if(num_samples == 1):
                
                zs = features
            else:
                z_mean = features[:, :self.num_latent]
                z_log_var = features[:, self.num_latent:]
                zs = sample_diagonal_MultiGauss(z_mean, z_log_var, num_samples)
            

        return zs