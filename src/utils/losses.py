import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Binary Cross-Entropy Loss
        bce = F.binary_cross_entropy_with_logits(input, target)
        
        # Dice Loss
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = input.size(0)  # Batch size
        channels = input.size(1)  # Number of channels
        
        input = input.view(num, channels, -1)  # Reshape to (batch, channels, flattened spatial)
        target = target.view(num, channels, -1)
        
        intersection = (input * target).sum(-1)  # Sum over spatial dimensions
        dice = (2. * intersection + smooth) / (input.sum(-1) + target.sum(-1) + smooth)  # Per channel
        
        dice = 1 - dice.mean()  # Average Dice loss over channels
        
        # Combined loss
        return 0.5 * bce + dice
