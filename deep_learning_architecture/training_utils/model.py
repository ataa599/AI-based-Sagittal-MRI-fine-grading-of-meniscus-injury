import torch.nn as nn
import torch
# =========================
# DenseNet Backbone Model
# =========================
class DenseNetSagittalModel(nn.Module):
    """
    DenseNet backbone with three heads selected by 'position':
      - position == 1 -> anterior head
      - position == 0 -> posterior head
      - position == 2 -> body head
    Expects input dict: {'image': Tensor[B,3,224,224], 'position': Tensor[B]}
    Returns: {'damage_logits': Tensor[B, num_classes]}
    """
    def __init__(self, num_classes=4, pretrained=False, densenet_variant='121'):
        super().__init__()
        from torchvision import models

        self.num_classes = num_classes
        variant = densenet_variant

        if variant not in {'121', '169', '201'}:
            raise ValueError("densenet_variant must be one of {'121','169','201'}")

        if variant == '121':
            try:
                weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
                base = models.densenet121(weights=weights)
            except Exception:
                base = models.densenet121(pretrained=pretrained)
        elif variant == '169':
            try:
                weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
                base = models.densenet169(weights=weights)
            except Exception:
                base = models.densenet169(pretrained=pretrained)
        else:  # '201'
            try:
                weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
                base = models.densenet201(weights=weights)
            except Exception:
                base = models.densenet201(pretrained=pretrained)

        self.base = base
        in_features = self.base.classifier.in_features  # 1024 for densenet121

        self.features = self.base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.anterior_classifier = nn.Linear(in_features, num_classes)
        self.posterior_classifier = nn.Linear(in_features, num_classes)
        self.body_classifier     = nn.Linear(in_features, num_classes)

    def forward(self, x):
        img = x['image']
        pos = x['position']

        feat_map = self.features(img)
        feat_map = self.relu(feat_map)
        pooled   = self.pool(feat_map)
        feat_vec = torch.flatten(pooled, 1)

        B = feat_vec.size(0)
        damage_logits = torch.zeros(B, self.num_classes, device=feat_vec.device)

        anterior_mask = (pos == 1)
        posterior_mask = (pos == 0)
        body_mask     = (pos == 2)

        if anterior_mask.any():
            damage_logits[anterior_mask] = self.anterior_classifier(feat_vec[anterior_mask])
        if posterior_mask.any():
            damage_logits[posterior_mask] = self.posterior_classifier(feat_vec[posterior_mask])
        if body_mask.any():
            damage_logits[body_mask] = self.body_classifier(feat_vec[body_mask])

        return {'damage_logits': damage_logits}