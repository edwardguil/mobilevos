# mobilevos/utils.py
import torch
import torch.nn.functional as F

def get_boundary_mask(mask, dilation=1):
    B, H, W = mask.shape
    mask = mask.float()
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(mask.unsqueeze(1), sobel_x, padding=1)
    grad_y = F.conv2d(mask.unsqueeze(1), sobel_y, padding=1)
    grad_mag = (grad_x.abs() + grad_y.abs())
    boundary = (grad_mag > 0).float()
    if dilation > 1:
        kernel_size = 2 * dilation + 1
        boundary = F.max_pool2d(boundary, kernel_size=kernel_size, stride=1, padding=dilation)
    return boundary.squeeze(1)