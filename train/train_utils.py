import torch
import torch.nn as nn
import kornia.augmentation as K

class GPUAugmentation(nn.Module):
    """Apply augmentations on GPU in batches"""
    def __init__(self, cutout_size):
        super().__init__()
        self.aug = nn.Sequential(
            K.CenterCrop((cutout_size, cutout_size)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=360.0),
        )
    
    def forward(self, x):
        return self.aug(x)