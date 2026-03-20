"""
DINOv2 Vision Encoder for extracting visual features from images.

This module defines the `DinoV2VisionEncoder` class, which wraps a pre-trained DINOv2 model to extract visual features from
input images. The vision encoder is meant to be used as the observation model.
"""

from typing import Literal

import torch
import torch.nn as nn

# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

FEATURE_KEYS = {"patch_tokens": "x_norm_patchtokens", "cls_token": "x_norm_clstoken"}


class DinoV2VisionEncoder(nn.Module):
    """
    DINOv2 Vision Encoder for extracting visual features from images.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        output_features: Literal["patch_tokens", "cls_token"] = "patch_tokens",
        frozen: bool = True,
    ) -> None:
        """
        Initializes the DINOv2 Vision Encoder.

        Args:
            model_name (str, optional): The name of the DINOv2 model to use. Defaults to "dinov2_vits14".
            output_features (Literal["patch_tokens", "cls_token"], optional): The type of features to output.
                If "patch_tokens", the encoder will return all patch embeddings (shape: batch_size, num_patches, emb_dim).
                If "cls_token", the encoder will return the CLS token embedding as a single patch (shape: batch_size, 1, emb_dim).
                The CLS token can be thought of as a summary representation of the whole image.
                Defaults to "patch_tokens".
            frozen (bool, optional): If True, the weights of the DINOv2 model will be frozen (not updated during training).
                Defaults to True.

        Raises:
            ValueError: If output_features is not one of "patch_tokens" or "cls_token".
        """
        super().__init__()
        self.name = model_name
        self.base_model: nn.Module = torch.hub.load("facebookresearch/dinov2", model_name)
        self.output_features = output_features
        if output_features not in FEATURE_KEYS:
            raise ValueError(
                f"Invalid argument for output_features: {output_features}. Must be either 'patch_tokens' or 'cls_token'."
            )
        self.patch_size = self.base_model.patch_size
        self.emb_dim = self.base_model.embed_dim

        # Freeze the DINOv2 model weights if specified
        if frozen:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DINOv2 Vision Encoder, which expects RGB 224 x 224 images as input.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Extracted visual features of shape (batch_size, num_patches, emb_dim).
        """
        embedding: torch.Tensor = self.base_model.forward_features(x)[
            FEATURE_KEYS[self.output_features]
        ]
        if self.output_features == "cls_token":
            # Add dummy patch dimension for CLS token (treat it as a single patch)
            embedding = embedding.unsqueeze(1)  # (batch_size, emb_dim) -> (batch_size, 1, emb_dim)
        return embedding
