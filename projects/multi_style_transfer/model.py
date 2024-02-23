import torch
from torch import nn

class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_styles, in_channels):
        super().__init__()
        self.net = nn.ModuleList([
            nn.InstanceNorm2d(in_channels, affine=True) for i in range(num_styles)
        ])

    def forward(self, x, style_id):
        return torch.stack([
            self.net[style_id[i]](x[i].unsqueeze(0)).squeeze_(0) for i in range(len(style_id))
        ])