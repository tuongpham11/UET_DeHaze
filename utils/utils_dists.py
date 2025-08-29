'''
DISTS metric implementation.
Reference: 
    Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. 
    "Image Quality Assessment: Unifying Structure and Texture Similarity." 
    TPAMI, 2020.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class L2Pool2d(nn.Module):
    '''L2 pooling for DISTS'''
    def __init__(self, kernel_size=3, stride=2, padding=1, same=False):
        super(L2Pool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.same = same

    def forward(self, x):
        if self.same:
            x = F.avg_pool2d(x.pow(2), self.kernel_size, self.stride, self.padding)
            return torch.sqrt(x + 1e-8)
        else:
            x = F.avg_pool2d(x.pow(2), self.kernel_size, self.stride, self.padding)
            return torch.sqrt(x + 1e-8)

class DISTS(nn.Module):
    '''
    DISTS metric
    Reference: 
        Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. 
        "Image Quality Assessment: Unifying Structure and Texture Similarity." 
        TPAMI, 2020.
    '''
    def __init__(self, pretrained=True):
        super(DISTS, self).__init__()
        
        vgg_pretrained = models.vgg16(pretrained=pretrained).features
        
        # VGG16 feature layers used for DISTS
        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential()
        self.stage5 = nn.Sequential()
        
        for i in range(0, 4):
            self.stage1.add_module(str(i), vgg_pretrained[i])
        self.stage2.add_module(str(4), L2Pool2d(kernel_size=3, stride=2, padding=1))
        for i in range(5, 9):
            self.stage2.add_module(str(i), vgg_pretrained[i])
        self.stage3.add_module(str(9), L2Pool2d(kernel_size=3, stride=2, padding=1))
        for i in range(10, 16):
            self.stage3.add_module(str(i), vgg_pretrained[i])
        self.stage4.add_module(str(16), L2Pool2d(kernel_size=3, stride=2, padding=1))
        for i in range(17, 23):
            self.stage4.add_module(str(i), vgg_pretrained[i])
        self.stage5.add_module(str(23), L2Pool2d(kernel_size=3, stride=2, padding=1))
        for i in range(24, 30):
            self.stage5.add_module(str(i), vgg_pretrained[i])
        
        # Weights for structure and texture terms
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, 1, 1, 1))
        
        # Weights for different VGG layers
        self.weights = nn.Parameter(torch.ones(5))
        
        # Register pre-trained parameters as buffers
        for param in self.parameters():
            param.requires_grad = False
            
        # Load pre-trained DISTS weights
        self.alpha.data.fill_(0.1)
        self.beta.data.fill_(0.1)
        self.weights.data = torch.from_numpy(np.array([0.1, 0.1, 0.1, 0.1, 0.1]))

    def forward_once(self, x):
        '''Forward pass to extract features'''
        h1 = self.stage1(x)
        h2 = self.stage2(h1)
        h3 = self.stage3(h2)
        h4 = self.stage4(h3)
        h5 = self.stage5(h4)
        return [x, h1, h2, h3, h4, h5]
    
    def compute_similarity(self, f1, f2):
        '''Compute structure and texture similarity'''
        # Structure similarity (cross-correlation)
        def structure_sim(x, y):
            x = x.view(x.shape[0], x.shape[1], -1)
            y = y.view(y.shape[0], y.shape[1], -1)
            
            # Normalize features
            x_mean = torch.mean(x, dim=2, keepdim=True)
            y_mean = torch.mean(y, dim=2, keepdim=True)
            x = x - x_mean
            y = y - y_mean
            
            x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
            y_norm = torch.norm(y, p=2, dim=2, keepdim=True)
            
            # Avoid division by zero
            x_norm[x_norm < 1e-8] = 1e-8
            y_norm[y_norm < 1e-8] = 1e-8
            
            x = x / x_norm
            y = y / y_norm
            
            # Compute cross correlation
            N = x.shape[2]
            cc = torch.bmm(x.transpose(1,2), y) / N  # Batch matrix multiplication
            cc = torch.mean(torch.diagonal(cc, dim1=1, dim2=2))
            return cc
        
        # Texture similarity (statistics matching)
        def texture_sim(x, y):
            x = x.view(x.shape[0], x.shape[1], -1)
            y = y.view(y.shape[0], y.shape[1], -1)
            
            # Mean and standard deviation
            x_mean = torch.mean(x, dim=2)
            y_mean = torch.mean(y, dim=2)
            x_std = torch.std(x, dim=2) + 1e-8  # Avoid division by zero
            y_std = torch.std(y, dim=2) + 1e-8
            
            # Compute mean similarity
            mean_sim = 1.0 - torch.mean(torch.abs(x_mean - y_mean) / (torch.abs(x_mean) + torch.abs(y_mean) + 1e-8))
            
            # Compute std similarity
            std_sim = 1.0 - torch.mean(torch.abs(x_std - y_std) / (x_std + y_std + 1e-8))
            
            return 0.5 * mean_sim + 0.5 * std_sim
        
        s_sim = structure_sim(f1, f2)
        t_sim = texture_sim(f1, f2)
        return self.alpha * s_sim + self.beta * t_sim
        
    def forward(self, x, y):
        '''Compute DISTS metric between two images'''
        assert x.shape == y.shape, f'Input shapes must match: {x.shape} vs {y.shape}'
        
        # Check if inputs are in the range [0, 1]
        if x.max() > 1.0 or y.max() > 1.0:
            x = x / 255.0
            y = y / 255.0
            
        # Extract VGG features
        feats_x = self.forward_once(x)
        feats_y = self.forward_once(y)
        
        # Compute similarity at each level
        score = 0
        for i in range(len(feats_x)):
            sim = self.compute_similarity(feats_x[i], feats_y[i])
            score += self.weights[i] * sim
            
        return 1.0 - score  # Convert similarity to distance (lower is better)

def calculate_dists(img1, img2):
    '''
    Calculate DISTS metric between two images
    Inputs:
        img1, img2: numpy arrays with values in [0, 255]
    '''
    # Handle CPU/GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize DISTS model
    dists_model = DISTS(pretrained=True).to(device)
    dists_model.eval()
    
    # Convert numpy arrays to tensors
    if isinstance(img1, np.ndarray):
        # Convert from HWC to NCHW format
        if img1.ndim == 3:  # HWC
            img1 = np.transpose(img1, (2, 0, 1))
            img2 = np.transpose(img2, (2, 0, 1))
        img1 = torch.from_numpy(img1).float().unsqueeze(0)
        img2 = torch.from_numpy(img2).float().unsqueeze(0)
        
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        score = dists_model(img1, img2)
        
    return score.item()
