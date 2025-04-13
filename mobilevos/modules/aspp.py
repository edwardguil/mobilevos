# mobilevos/modules/aspp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.atrous_block1(x)
        block6 = self.atrous_block6(x)
        block12 = self.atrous_block12(x)
        block18 = self.atrous_block18(x)
        x = torch.cat([block1, block6, block12, block18], dim=1)
        return self.conv_1x1_output(x)