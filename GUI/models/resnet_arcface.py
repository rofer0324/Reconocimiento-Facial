import torch
import torch.nn as nn
import torch.nn.functional as F
from models.arcface import ArcFace
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights


class ResNetArcModel(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="resnet50",
        embedding_size=512,
        arcface_margin=0.5,
        arcface_scale=30.0,
        arcface_easy_margin=False,
    ):
        """
        Wraps a ResNet backbone and replaces final layer with embedding + ArcFace.

        Args:
            num_classes: Number of output classes.
            backbone: choose resnet version 'resnet18', 'resnet50'
            embedding_size: Output dimension of the embedding before classification.
        """

        super(ResNetArcModel, self).__init__()

        # Load a PRETRAINED ResNet and remove the original classifier
        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            resnet = resnet50(weights=weights)
        elif backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            resnet = resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()  # type: ignore

        self.backbone = resnet

        # Project backbone output to lower-dim embedding
        self.embedding = nn.Linear(in_features, embedding_size)

        # ArcFace classification head
        self.arcface = ArcFace(
            embedding_size,
            num_classes,
            s=arcface_scale,
            m=arcface_margin,
            easy_margin=arcface_easy_margin,
        )

    def forward(self, x, labels=None):
        """
        Forward pass through the full model.

        Args
            x: Input image tensor [B, C, H, W]
            labels: Target labels [B], required for training
        Returns:
            If labels are provided: ArcFace logits (used in training).
            If not: Raw embeddings (used in inference).
        """

        x = self.backbone(x)  # extract features from ResNet

        x = self.embedding(x)  # project to embedding space

        if labels is not None:
            logits = self.arcface(x, labels)  # compute logits with arc margin
            return logits

        return x  # inference mode: return embeddings
