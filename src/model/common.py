import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


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
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
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
# --- START: Our Custom GSE Code ---
# -------------------------------------------------------------------------

def gse(x_in, layer, median=True):
    """
    Performs Geometric Self-Ensemble on the input x_in using the provided layer.
    (This is your function from Cell 2)
    """
    # 1. Generate 8 transformations
    # We pad x_in to avoid border artifacts after rotation and flips
    # This assumes kernel_size 3 (pad=1). For larger kernels, this might need adjustment.
    x_in_padded = F.pad(x_in, (1, 1, 1, 1), mode='replicate')

    x_list = []
    for i in range(8):
        x_aug = x_in_padded.rot90(i, [2, 3])
        if i % 2 == 1:
            x_aug = x_aug.flip(3)
        x_list.append(x_aug)

    # 2. Apply the layer to each transformation
    # We process in a batch to be efficient on the GPU
    x_batch = torch.cat(x_list, dim=0)
    y_batch = layer(x_batch)
    y_list = y_batch.chunk(8, dim=0)

    # 3. Invert the transformations
    y_inv_list = []
    for i in range(8):
        y_aug = y_list[i]
        if i % 2 == 1:
            y_aug = y_aug.flip(3)
        y_inv = y_aug.rot90(-i, [2, 3])

        # Un-pad the output to match original size
        y_inv_unpadded = y_inv[:, :, 1:-1, 1:-1]
        y_inv_list.append(y_inv_unpadded)

    # 4. Aggregate the results
    y_all = torch.stack(y_inv_list, dim=0)
    if median:
        y_agg = y_all.median(dim=0).values
    else:
        y_agg = y_all.mean(dim=0)

    return y_agg


class GSEWrapper(torch.nn.Module):
    """
    A wrapper to apply GSE to a layer, with controls for scaling.
    This reads 'scale_factor' from the arguments.
    """

    def __init__(self, layer, median=True, scale_factor=1.0):
        super(GSEWrapper, self).__init__()
        self.layer = layer
        self.median = median
        self.scale_factor = scale_factor  # This is our new control

    def forward(self, x):
        # 1. Get the output of the GSE-wrapped layer
        #    Note: 'layer' is a ResBlock, so y_gse = x + residual_gse
        y_gse = gse(x, self.layer, self.median)

        # 2. Isolate the GSE-based residual
        #    residual_gse = (x + residual_gse) - x
        residual_gse = y_gse - x

        # 3. Apply the scaling factor (lambda) to the residual
        scaled_residual = residual_gse * self.scale_factor

        # 4. Return the new, scaled output
        #    This is x + (lambda * residual_gse)
        return x + scaled_residual

# --- END: Our Custom GSE Code ---