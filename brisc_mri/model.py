"""Notebook-compatible ResNet34 / FiLM multi-task U-Net."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class DecoderBlock(nn.Module):
    """Upsamples feature maps by 2x and fuses them with encoder skip connections."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Enables MC-Dropout pixel uncertainty later
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PlaneConditionedMultiTaskUNet(nn.Module):
    def __init__(self, num_classes=4, num_planes=3, pretrained=False):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        # Encoder stages (256 -> 128 -> 64 -> 32 -> 16 -> 8)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64 ch, 128x128
        self.pool = backbone.maxpool                                            # 64 ch, 64x64
        self.enc1 = backbone.layer1                                             # 64 ch, 64x64
        self.enc2 = backbone.layer2                                             # 128 ch, 32x32
        self.enc3 = backbone.layer3                                             # 256 ch, 16x16
        self.enc4 = backbone.layer4                                             # 512 ch, 8x8

        # FiLM Plane Conditioning at the 512-channel bottleneck
        self.plane_emb = nn.Embedding(num_planes, 64)
        self.film = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512 * 2)  # Outputs gamma (512) and beta (512)
        )

        # Head 1: 4-Class Tumor Classifier
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Head 2: U-Net Segmentation Decoder (8 -> 16 -> 32 -> 64 -> 128 -> 256)
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x, plane_idx):
        # 1. Encode MRI
        s0 = self.stem(x)            # [B, 64, 128, 128]
        s1 = self.enc1(self.pool(s0)) # [B, 64, 64, 64]
        s2 = self.enc2(s1)           # [B, 128, 32, 32]
        s3 = self.enc3(s2)           # [B, 256, 16, 16]
        bottleneck = self.enc4(s3)   # [B, 512, 8, 8]

        # 2. Condition bottleneck on Anatomical Plane (FiLM modulation)
        p_vec = self.plane_emb(plane_idx)
        gamma, beta = self.film(p_vec).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        cond_feat = bottleneck * (1.0 + gamma) + beta

        # 3. Predict Class Logits & Segmentation Mask Logits simultaneously
        cls_logits = self.cls_head(cond_feat)
        d4 = self.dec4(cond_feat, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        seg_logits = self.final_up(d1)

        return cls_logits, seg_logits
