import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

# vp 部分 (SSM)
class AttributeNet(nn.Module):
    def __init__(self, layers=5, patch_size=8, channels=3):
        super(AttributeNet, self).__init__()
        self.layers = layers
        self.patch_size = patch_size
        self.channels = channels

        self.pooling = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        if self.layers == 5 and self.channels == 3:
            self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)
        elif self.layers == 6:
            self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            self.bn5 = nn.BatchNorm2d(128)
            self.relu5 = nn.ReLU(inplace=True)

            if self.channels == 3:
                self.conv6 = nn.Conv2d(128, 3, 3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        if self.patch_size in [2, 4, 8, 16, 32]:
            y = self.pooling(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        if self.patch_size in [4, 8, 16, 32]:
            y = self.pooling(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        if self.patch_size in [8, 16, 32]:
            y = self.pooling(y)

        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        if self.patch_size in [16, 32]:
            y = self.pooling(y)

        if self.layers == 6:
            y = self.conv5(y)
            y = self.bn5(y)
            y = self.relu5(y)
            if self.patch_size == 32:
                y = self.pooling(y)

        if self.channels == 3:
            y = self.conv6(y)
        elif self.channels == 1:
            y = torch.mean(y, dim=1, keepdim=True)  # shape: [B, 1, H, W]

        return y


class InstancewiseVisualPrompt(nn.Module):
    def __init__(self, size, layers=5, patch_size=8, channels=3):
        super(InstancewiseVisualPrompt, self).__init__()

        if layers not in [5, 6]:
            raise ValueError("Only 5 or 6 layers supported.")
        if patch_size not in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Unsupported patch size.")
        if channels not in [1, 3]:
            raise ValueError("channels must be 1 or 3.")
        if patch_size == 32 and layers != 6:
            raise ValueError("Patch size 32 requires 6-layer network.")

        self.size = size
        self.channels = channels
        self.patch_size = patch_size
        self.patch_num = size // patch_size

        self.priority = AttributeNet(layers, patch_size, channels)
        self.program = nn.Parameter(torch.zeros(3, size, size))

    def forward(self, x):
        B = x.size(0)
        attention = self.priority(x).view(B, self.channels, self.patch_num * self.patch_num, 1)
        attention = attention.expand(-1, 3, -1, self.patch_size * self.patch_size)
        attention = attention.view(B, 3, self.patch_num, self.patch_num, self.patch_size, self.patch_size).transpose(3, 4)
        attention = attention.reshape(B, 3, self.size, self.size)
        return x + self.program.unsqueeze(0) * attention


class DummyVisualPrompt(nn.Module):
    def forward(self, x):
        return x



# =============== 基础：正则（可选） ===============
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


# watermarking
class WatermarkingVR(nn.Module):
    def __init__(self, size, pad):
        super().__init__()
        self.size = size
        self.program = nn.Parameter(torch.zeros(3, size, size))
        self.alpha = nn.Parameter(torch.ones(3,1,1)*0.5)
        if size > 2*pad:
            # 边框区域=1，中心=0
            mask = torch.zeros(3, size-2*pad, size-2*pad)
            self.register_buffer("mask", F.pad(mask, [pad, pad, pad, pad], value=1))
        else:
            self.register_buffer("mask", torch.ones(3, size, size))

    def forward(self, x):
        delta = self.alpha.to(x.dtype) * torch.tanh(self.program.to(x.dtype)) * self.mask.to(x.dtype)
        return x + delta

# padding watermarking  
class PaddingVR(nn.Module):
    def __init__(self, out_size, mask_np, init='zero', normalize=None):
        super().__init__()
        assert mask_np.shape[0] == mask_np.shape[1]
        in_size = mask_np.shape[0]
        self.out_size = out_size
        self.normalize = normalize
        if init == "zero":
            self.program = nn.Parameter(torch.zeros(3, out_size, out_size))
        elif init == "randn":
            self.program = nn.Parameter(torch.randn(3, out_size, out_size))
        else:
            raise ValueError("init must be 'zero' or 'randn'")
        self.alpha = nn.Parameter(torch.ones(3,1,1)*0.5)

        self.l_pad = int((out_size - in_size + 1) / 2)
        self.r_pad = int((out_size - in_size) / 2)
        # 传入的 mask_np 控制原图区域；pad 出来的边缘默认允许扰动（value=1）
        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).repeat(3,1,1)
        self.register_buffer("mask", F.pad(mask, (self.l_pad,self.r_pad,self.l_pad,self.r_pad), value=1))

    def forward(self, x):
        x = F.pad(x, (self.l_pad,self.r_pad,self.l_pad,self.r_pad), value=0)  # pad 到 out_size
        delta = self.alpha.to(x.dtype) * torch.tanh(self.program.to(x.dtype)) * self.mask.to(x.dtype)
        x = x + delta
        if self.normalize is not None:
            x = self.normalize(x)
        return x



def build_visual_prompt(
    image_size=224,
    layers=5,
    patch_size=8,
    channels=3,
    mode: str = "instance",     # 'instance' | 'watermark' | 'padding' | 'none'
    pad: int = 30,
    padding_in_size: int | None = None,  # 给 padding 模式用：原图尺寸（默认 image_size-2*pad）
    padding_mask_value: float = 0.0,     # 给 padding 模式用：原图区域是否允许扰动（0=不允许，1=允许）
    normalize=None,                      # 给 padding 模式的可选 Normalize
) -> nn.Module:
    if mode == "none":
        print("[Visual Prompt] Disabled")
        return DummyVisualPrompt()

    if mode == "instance":
        print(f"[VP] Instancewise: size={image_size}, layers={layers}, patch={patch_size}, ch={channels}")
        return InstancewiseVisualPrompt(image_size, layers, patch_size, channels)

    if mode == "watermark":
        print(f"[VP] Watermarking: size={image_size}, pad={pad}")
        return WatermarkingVR(image_size, pad)

    if mode == "padding":
        # 推断 padding 内部的原图尺寸（默认中心不加扰动，只有边缘 padding 区域允许）
        if padding_in_size is None:
            padding_in_size = max(1, image_size - 2*pad)
        mask_np = np.full((padding_in_size, padding_in_size), fill_value=padding_mask_value, dtype=np.float32)
        print(f"[VP] Padding: out_size={image_size}, in_size={padding_in_size}, pad={pad}")
        return PaddingVR(image_size, mask_np, init='zero', normalize=normalize)

    raise ValueError(f"Unknown vp mode: {mode}")





    
            
