# src/features/resnet_patch_extractor.py
"""
ResNet-based patch encoders for writer retrieval.

Includes:
- ResNetSmall: Custom tiny ResNet-20-like for 32x32 patches
- ResNet18Encoder: Pretrained ResNet-18 adapted for grayscale
- ResNet34Encoder: Pretrained ResNet-34 adapted for grayscale
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNetSmall(nn.Module):
    """
    A small ResNet-like encoder for 32x32 grayscale patches.
    Roughly analogous to a tiny ResNet-20.
    """

    def __init__(self, block=BasicBlock, num_blocks=(3, 3, 3),
                 base_channels=32, emb_dim=128):
        super().__init__()
        self.in_planes = base_channels

        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, num_blocks[2], stride=2)

        # global average pooling to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4 * block.expansion, emb_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = (stride,) + (1,) * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 1, H, W]
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)   # maybe 32x32
        out = self.layer2(out)   # maybe 16x16
        out = self.layer3(out)   # maybe 8x8
        out = self.avgpool(out)  # [B, C, 1, 1]
        out = torch.flatten(out, 1)
        emb = self.fc(out)       # [B, emb_dim]
        emb = F.normalize(emb, dim=1)  # L2 normalization (common for retrieval)
        return emb


class ResNet18Encoder(nn.Module):
    """
    Pretrained ResNet-18 adapted for grayscale input.
    
    The first conv layer is modified to accept 1-channel input.
    Weights are initialized by averaging the 3-channel pretrained weights.
    
    This typically gives better features than training from scratch,
    especially for small datasets like CVL (27 writers).
    """
    
    def __init__(self, emb_dim=128, pretrained=True, freeze_bn=False):
        super().__init__()
        
        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Modify first conv layer for grayscale (1 channel instead of 3)
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize with averaged weights from RGB channels
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        
        # Copy other layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # New embedding layer
        self.fc = nn.Linear(512, emb_dim)
        
        # Optionally freeze BatchNorm layers (helps with small batches)
        if freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        # x: [B, 1, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.fc(x)
        emb = F.normalize(emb, dim=1)
        return emb


class ResNet34Encoder(nn.Module):
    """
    Pretrained ResNet-34 adapted for grayscale input.
    
    Deeper than ResNet-18, may perform better for complex datasets.
    """
    
    def __init__(self, emb_dim=128, pretrained=True, freeze_bn=False):
        super().__init__()
        
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)
        
        # Modify first conv layer for grayscale
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.fc = nn.Linear(512, emb_dim)
        
        if freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.fc(x)
        emb = F.normalize(emb, dim=1)
        return emb


class ResNetSmallDeep(nn.Module):
    """
    A deeper custom ResNet for 32x32 patches (ResNet-56-like).
    Similar to the reference icdar23 implementation.
    
    This has more capacity than ResNetSmall but still designed for small patches.
    """
    
    def __init__(self, block=BasicBlock, num_blocks=(9, 9, 9),
                 base_channels=32, emb_dim=128):
        super().__init__()
        self.in_planes = base_channels
        
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4 * block.expansion, emb_dim)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = (stride,) + (1,) * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        emb = self.fc(out)
        emb = F.normalize(emb, dim=1)
        return emb


def create_resnet_patch_encoder(emb_dim=128, backbone='small', pretrained=True, 
                                 freeze_bn=False, debug=False):
    """
    Factory to create ResNet encoder for patches.
    
    This encoder produces LOCAL PATCH EMBEDDINGS that can be aggregated
    into a GLOBAL DESCRIPTOR using NetVLAD/NetRVLAD/GeM pooling.
    
    Args:
        emb_dim: Embedding dimension (typically 64-256)
        backbone: One of 'small', 'small_deep', 'resnet18', 'resnet34'
        pretrained: Use ImageNet pretrained weights (for resnet18/34)
        freeze_bn: Freeze BatchNorm layers (helps with small batches)
        debug: Enable debug logging
    
    Returns:
        ResNet encoder module
    """
    import logging
    logger = logging.getLogger(__name__)
    
    backbone_map = {
        'small': lambda: ResNetSmall(emb_dim=emb_dim),
        'small_deep': lambda: ResNetSmallDeep(emb_dim=emb_dim),
        'resnet18': lambda: ResNet18Encoder(emb_dim=emb_dim, pretrained=pretrained, freeze_bn=freeze_bn),
        'resnet34': lambda: ResNet34Encoder(emb_dim=emb_dim, pretrained=pretrained, freeze_bn=freeze_bn),
    }
    
    if backbone not in backbone_map:
        raise ValueError(f"Unknown backbone: {backbone}. Available: {list(backbone_map.keys())}")
    
    model = backbone_map[backbone]()
    
    if debug:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.debug(f'[DEBUG] Created {backbone} encoder: emb_dim={emb_dim}, '
                    f'params={num_params:,}, pretrained={pretrained}')
    
    return model
