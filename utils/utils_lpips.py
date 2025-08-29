'''
LPIPS metric implementation.
Reference:
    Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).
    "The unreasonable effectiveness of deep features as a perceptual metric."
    In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR).
'''

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class LPIPS(nn.Module):
    '''
    LPIPS metric for perceptual image quality assessment
    '''
    def __init__(self, net='alex', use_gpu=True):
        '''
        Parameters:
            net: 'alex' (default) | 'vgg' | 'squeeze' - backbone network
            use_gpu: whether to use GPU
        '''
        super(LPIPS, self).__init__()
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Load pretrained network
        if net == 'alex':
            self.net = models.alexnet(pretrained=True).features.to(self.device)
            self.layer_indices = [0, 3, 6, 8, 10]  # Conv layers from AlexNet
            self.weights = [0.0, 0.0, 0.2, 0.3, 0.5]  # Custom weights for different layers
        elif net == 'vgg':
            self.net = models.vgg16(pretrained=True).features.to(self.device)
            self.layer_indices = [0, 4, 9, 16, 23]  # Conv layers from VGG16
            self.weights = [0.0, 0.0, 0.2, 0.3, 0.5]  # Custom weights
        elif net == 'squeeze':
            self.net = models.squeezenet1_1(pretrained=True).features.to(self.device)
            self.layer_indices = [0, 4, 7, 9, 12]  # Conv layers from SqueezeNet
            self.weights = [0.0, 0.0, 0.2, 0.3, 0.5]  # Custom weights
        else:
            raise ValueError(f"Network '{net}' not implemented. Use 'alex', 'vgg', or 'squeeze'.")
        
        # Set model to evaluation mode
        self.net.eval()
        
        # Freeze network parameters
        for param in self.net.parameters():
            param.requires_grad = False
            
        # Normalization for network inputs
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def normalize(self, tensor):
        '''Normalize input images to match ImageNet statistics'''
        return (tensor - self.mean) / self.std
    
    def get_features(self, x):
        '''Extract features from different layers of the network'''
        features = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features
    
    def spatial_average(self, x):
        '''Average across spatial dimensions'''
        return x.mean([2, 3], keepdim=True)
    
    def forward(self, x, y):
        '''
        Calculate LPIPS distance between two images or batches of images
        x, y: Image tensors with shape (N, C, H, W) in range [0, 1]
        '''
        if x.shape != y.shape:
            raise ValueError(f'Input shapes must match: {x.shape} vs {y.shape}')
        
        # Check if inputs are in the range [0, 1]
        if x.max() > 1.0 or y.max() > 1.0:
            x = x / 255.0
            y = y / 255.0
        
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Normalize images
        x = self.normalize(x)
        y = self.normalize(y)
        
        # Extract features
        feats_x = self.get_features(x)
        feats_y = self.get_features(y)
        
        # Calculate distances
        distances = []
        for fx, fy, weight in zip(feats_x, feats_y, self.weights):
            # Normalize features
            n_fx = fx / (torch.sqrt(torch.sum(fx**2, dim=1, keepdim=True)) + 1e-10)
            n_fy = fy / (torch.sqrt(torch.sum(fy**2, dim=1, keepdim=True)) + 1e-10)
            
            # Calculate distance
            dist = torch.mean((n_fx - n_fy)**2, dim=1)
            
            # Spatial pooling
            dist = self.spatial_average(dist)
            distances.append(dist * weight)
        
        # Sum weighted distances
        lpips_value = sum(distances).squeeze()
        
        return lpips_value


def calculate_lpips(img1, img2, net='alex'):
    '''
    Calculate LPIPS metric between two images
    Inputs:
        img1, img2: numpy arrays with values in [0, 255]
        net: 'alex' (default) | 'vgg' | 'squeeze' - backbone network
    '''
    # Handle CPU/GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize LPIPS model
    lpips_model = LPIPS(net=net, use_gpu=device.type=='cuda')
    
    # Convert numpy arrays to tensors
    if isinstance(img1, np.ndarray):
        # Convert from HWC to NCHW format
        if img1.ndim == 3:  # HWC
            img1 = np.transpose(img1, (2, 0, 1))
            img2 = np.transpose(img2, (2, 0, 1))
        img1 = torch.from_numpy(img1).float().unsqueeze(0)
        img2 = torch.from_numpy(img2).float().unsqueeze(0)
    
    with torch.no_grad():
        score = lpips_model(img1, img2)
    
    return score.cpu().item()
