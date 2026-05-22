import torch
import torch.nn as nn
from functools import partial
from resnet import BasicBlock, Bottleneck, conv3x3x3, downsample_basic_block


class FPClassifier(nn.Module):
    """3D ResNet backbone + classification head for FP reduction.

    Backbone architecture is identical to MedicalNet's 3D ResNet so pretrained
    weights can be loaded directly (after stripping 'module.' prefix).

    Forward path: conv1 → bn1 → relu → maxpool → layer1-4 → GAP → classifier → sigmoid
    """

    def __init__(self, block, layers, in_channels=1, dropout=0.5):
        self.inplanes = 64
        super().__init__()

        # ---- Stem (identical to MedicalNet) ----
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # ---- ResNet stages (identical to MedicalNet) ----
        self.layer1 = self._make_layer(block, 64, layers[0], 'A')
        self.layer2 = self._make_layer(block, 128, layers[1], 'A', stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], 'A', stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], 'A', stride=1, dilation=4)

        # ---- Classification head ----
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * block.expansion, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=False)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers_list = []
        layers_list.append(block(self.inplanes, planes, stride=stride,
                                 dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def load_medicalnet(self, checkpoint_path):
        """Load MedicalNet pretrained backbone weights.

        Strips 'module.' prefix (from DataParallel), skips segmentation-head
        keys that don't exist in this model, and leaves the classifier head
        randomly initialized.

        Returns list of loaded backbone keys for verification.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        cleaned = {}
        skipped = []
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            if 'conv_seg' in new_key:
                skipped.append(new_key)
                continue
            cleaned[new_key] = v

        missing, unexpected = self.load_state_dict(cleaned, strict=False)

        backbone_missing = [k for k in missing if 'classifier' not in k]
        if backbone_missing:
            raise RuntimeError(f"Backbone key mismatch — not MedicalNet-compatible: {backbone_missing}")

        return {
            'loaded': len(cleaned),
            'skipped_seg': len(skipped),
            'unexpected': unexpected,
        }


def build_fp_classifier(backbone='resnet18', pretrained=None, dropout=0.5):
    """Factory: build FPClassifier with optional MedicalNet pretrained weights.

    Parameters
    ----------
    backbone : str
        'resnet18' or 'resnet34'.
    pretrained : str or None
        Path to MedicalNet .pth checkpoint. None = random init (Strategy A).
    dropout : float
        Dropout rate for classification head.

    Returns
    -------
    FPClassifier
    """
    configs = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
    }
    if backbone not in configs:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(configs.keys())}")

    block, layers = configs[backbone]
    model = FPClassifier(block, layers, dropout=dropout)

    if pretrained is not None:
        info = model.load_medicalnet(pretrained)
        print(f"[{backbone}] MedicalNet weights loaded: {info['loaded']} backbone keys, "
              f"skipped {info['skipped_seg']} seg-head keys")

    return model
