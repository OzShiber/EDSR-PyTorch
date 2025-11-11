import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# -------------------------------------------------------------------------
# --- START: Custom GSE Code ---
# -------------------------------------------------------------------------

def gse(x_in, layer, median=True):
    """
    Performs Geometric Self-Ensemble on the input x_in using the provided layer.
    This version correctly handles non-square images by splitting the
    augmentations into two groups and processing them in separate batches.
    """
    
    # We still use 1px padding for convolution border artifacts
    x_in_padded = F.pad(x_in, (1, 1, 1, 1), mode='replicate')
    
    # Group 1: Shape-preserving transforms (Identity, V-Flip, H-Flip, 180-Rot)
    # 0: identity
    # 2: vflip
    # 4: hflip
    # 6: vflip + hflip (180 rot)
    x_list_group1 = []
    x_list_group1.append(x_in_padded)                     # 0
    x_list_group1.append(x_in_padded.flip(2))             # 2
    x_list_group1.append(x_in_padded.flip(3))             # 4
    x_list_group1.append(x_in_padded.flip(2).flip(3))     # 6
    
    x_batch_group1 = torch.cat(x_list_group1, dim=0)
    y_batch_group1 = layer(x_batch_group1)
    y_list_group1 = y_batch_group1.chunk(4, dim=0)
    
    # Group 2: Shape-changing transforms (90-Rot, 270-Rot, and flips)
    # 1: rot90
    # 3: rot90 + vflip
    # 5: rot90 + hflip
    # 7: rot90 + vflip + hflip
    x_list_group2 = []
    x_list_group2.append(x_in_padded.rot90(1, [2, 3]))            # 1
    x_list_group2.append(x_in_padded.rot90(1, [2, 3]).flip(2))    # 3
    x_list_group2.append(x_in_padded.rot90(1, [2, 3]).flip(3))    # 5
    x_list_group2.append(x_in_padded.rot90(1, [2, 3]).flip(2).flip(3)) # 7

    x_batch_group2 = torch.cat(x_list_group2, dim=0)
    y_batch_group2 = layer(x_batch_group2)
    y_list_group2 = y_batch_group2.chunk(4, dim=0)

    # --- Invert Transformations ---
    # We re-assemble the list in its original 0-7 order
    y_inv_list = [None] * 8
    
    # Invert Group 1
    y_inv_list[0] = y_list_group1[0]
    y_inv_list[2] = y_list_group1[1].flip(2)
    y_inv_list[4] = y_list_group1[2].flip(3)
    y_inv_list[6] = y_list_group1[3].flip(2).flip(3)

    # Invert Group 2 (note the -1 rotation)
    y_inv_list[1] = y_list_group2[0].rot90(-1, [2, 3])
    y_inv_list[3] = y_list_group2[1].flip(2).rot90(-1, [2, 3])
    y_inv_list[5] = y_list_group2[2].flip(3).rot90(-1, [2, 3])
    y_inv_list[7] = y_list_group2[3].flip(2).flip(3).rot90(-1, [2, 3])

    # --- Aggregate Results ---
    # Un-pad the 1px border from all 8 tensors
    y_unpadded_list = []
    for y_inv in y_inv_list:
        y_unpadded_list.append(y_inv[:, :, 1:-1, 1:-1])

    y_all = torch.stack(y_unpadded_list, dim=0)
    
    if median:
        y_agg = y_all.median(dim=0).values
    else:
        y_agg = y_all.mean(dim=0)
        
    return y_agg

class GSEWrapper(torch.nn.Module):
    """
    A wrapper to apply GSE to a layer, with controls for scaling.
    """
    def __init__(self, layer, median=True, scale_factor=1.0):
        super(GSEWrapper, self).__init__()
        self.layer = layer
        self.median = median
        self.scale_factor = scale_factor # This is our new control

    def forward(self, x):
        # 1. Get the output of the GSE-wrapped layer
        y_gse = gse(x, self.layer, self.median)
        
        # 2. Isolate the GSE-based residual
        residual_gse = y_gse - x 
        
        # 3. Apply the scaling factor (lambda) to the residual
        scaled_residual = residual_gse * self.scale_factor
        
        # 4. Return the new, scaled output
        return x + scaled_residual

# --- END: Custom GSE Code ---
