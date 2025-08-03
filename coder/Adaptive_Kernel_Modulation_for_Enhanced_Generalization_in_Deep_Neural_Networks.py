import torch
import torch.nn as nn
import torch.nn.functional as F

class AKMConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AKMConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.r = nn.Parameter(torch.randn(1)) # Receptive Field Modulation
        self.s = nn.Parameter(torch.randn(1)) # Feature Selectivity Modulation

    def forward(self, x):
        z = self.conv(x)
        z_prime = self.gaussian_kernel(z, self.r) # Apply receptive field modulation
        y = torch.sigmoid(self.s) * z_prime       # Apply feature selectivity modulation
        return y

    def gaussian_kernel(self, x, r):
        # Implement Gaussian smoothing (Simplified example for demonstration)
        sigma = torch.abs(r)  # Ensure sigma is positive
        kernel_size = int(3 * sigma) * 2 + 1 # Kernel size must be odd
        if kernel_size > 1:
            kernel = self.gaussian_filter(kernel_size, sigma)
            kernel = kernel.to(x.device)
            padding = kernel_size // 2
            x = F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
        return x
    
    def gaussian_filter(self, kernel_size, sigma):
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel = gauss / gauss.sum()
        return kernel.view(1, 1, -1)

# Example Usage:
in_channels = 3
out_channels = 16
kernel_size = 3
input_tensor = torch.randn(1, in_channels, 32, 32) # Batch size 1, 3x32x32 image
akm_conv = AKMConv2d(in_channels, out_channels, kernel_size)
output_tensor = akm_conv(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")