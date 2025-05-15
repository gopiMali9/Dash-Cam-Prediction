import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision.models.video.swin_transformer import swin3d_s
from pytorchvideo.models.hub import slowfast_r50
    

def tsm(tensor, duration, dilation=1):
    # [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # [N, T, C, H, W]
    shift_size = size[1] // 8
    pre_tensor, post_tensor, peri_tensor = tensor.split([shift_size, shift_size, 6 * shift_size], dim=2)
    pre_tensor = F.pad(pre_tensor, (0, 0, 0, 0, 0, 0, 0, dilation))[:, dilation:, ...]
    post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, dilation, 0))[:, :-dilation, ...]
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)


def add_tsm_to_module(obj, duration, dilation=1):
    orig_forward = obj.forward
    def updated_forward(*args, **kwargs):
        a = (tsm(args[0], duration=duration, dilation=dilation), ) + args[1:]
        return orig_forward(*a, **kwargs)
    obj.forward = updated_forward
    return obj



class TsmModel(nn.Module):
    def __init__(self, base_model_name,nb_frames,in_chans,dilations: List[int],add_tsm_to_all_blocks=False, pretrained=True):
        super().__init__()

        self.nb_frames = nb_frames
        self.base_model_name = base_model_name
        self.in_chans = in_chans

        self.backbone = timm.create_model(base_model_name,
                                          in_chans = self.in_chans,
                                          features_only=True,
                                          pretrained=False,
                                          num_classes=500)
        duration = nb_frames
        for block_num, block in enumerate(self.backbone.blocks):
            if dilations[block_num] > 0:
                for sub_block in block:
                    add_tsm_to_module(sub_block.conv_pw, duration, dilation=dilations[block_num])

        self.mlp = mlp = nn.Sequential(nn.Linear(8000, 1024),
                                        nn.LayerNorm(1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(1024, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(256, 1))

    def forward(self, inputs):
    
        B, T, C, H, W = inputs.shape
        x = inputs.view(B * T, C, H, W)
        x = self.backbone(x)
        x = x.reshape(B,-1)
        x = self.mlp(x)
        return x
    

class VideoSwinModel(nn.Module):
    def __init__(self, pretrained=True,freeze_weight=True):
        super(VideoSwinModel, self).__init__()
        self.backbone = swin3d_s(weights=pretrained)
        if freeze_weight:
            print('Weight Freezed')
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.patch_embed.proj = nn.Conv3d(4, 96, kernel_size=(2, 4, 4), stride=(2, 4, 4))

        self.mlp = nn.Sequential(
            nn.Linear(400, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)
        return x

class SlowFastRGBD(nn.Module):
    def __init__(self, pretrained=True, alpha=4):
        super().__init__()
        self.base = slowfast_r50(pretrained=pretrained)
        self._update_stem(self.base.blocks[0].multipathway_blocks[0], 4)
        self._update_stem(self.base.blocks[0].multipathway_blocks[1], 4)
        in_features = self.base.blocks[-1].proj.in_features
        self.base.blocks[-1].proj = nn.Linear(in_features, 1)
        self.alpha = alpha

    def _update_stem(self, stem, in_channels):
        old_conv = stem.conv
        new_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias= False
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            if in_channels > 3:
                nn.init.kaiming_normal_(new_conv.weight[:, 3:])
        stem.conv = new_conv

    def pack_pathway_input(self, x):
        # [batch, 4, T, H, W]
        fast_pathway = x
        slow_pathway = x[:, :, ::self.alpha, :, :]
        return [slow_pathway, fast_pathway]

    def forward(self, x):
        x = self.pack_pathway_input(x)
        return self.base(x)  # Output: (batch, 1)