"""
Transition Model for modeling the dynamics of the environment in the latent space.
Adopts a ViT architecture with a custom fixed block-causal attention mask to ensure proper temporal modeling for a
sequence of patches across multiple frames.

Code largely adapted from https://github.com/gaoyuezhou/dino_wm/blob/main/models/vit.py
"""

import torch
from einops import rearrange
from torch import nn


def generate_block_causal_mask_matrix(num_patches: int, num_frames: int) -> torch.Tensor:
    """
    Creates a fixed block-causal mask matrix for a sequence of patches across multiple frames (time steps).
    In other words, the expected sequence length is (num_frames * num_patches),
    and the mask ensures that each patch can only attend to patches from the same or previous frames,
    not future frames.

    Args:
        num_patches (int): The number of patches per frame.
        num_frames (int): The number of frames.

    Returns:
        torch.Tensor: The block-causal mask matrix of shape (1, 1, num_patches * num_frames, num_patches * num_frames).
    """
    zeros = torch.zeros(num_patches, num_patches)
    ones = torch.ones(num_patches, num_patches)
    rows = []
    for i in range(num_frames):
        # For each timestep, create a row showing which previous timesteps are visible
        row = torch.cat([ones] * (i + 1) + [zeros] * (num_frames - i - 1), dim=1)
        rows.append(row)
    # Concatenate rows to form the full mask matrix
    mask = (
        torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    )  # Shape: (1, 1, num_frames * num_patches, num_frames * num_patches)
    return mask


class FeedForward(nn.Module):
    """
    A simple feedforward network to be used within the Transformer blocks, consisting of two linear layers
    with a GELU activation and dropout in between.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        """
        Initializes the feedforward network.

        Args:
            dim (int): The input dimension.
            hidden_dim (int): The dimension of the hidden layer.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim) after processing through the feedforward network.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Attention mechanism that computes multi-head self-attention with a fixed block-causal mask for a fixed length sequence of
    patches across multiple frames. The mask ensures that each patch can only attend to patches from the same or previous frames,
    not future frames.
    """

    def __init__(
        self,
        num_patches: int,
        num_frames: int,
        dim_input: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the multi-head self-attention mechanism with a fixed block-causal mask for temporal sequences of patches.

        Args:
            num_patches (int): The number of patches per frame.
            num_frames (int): The number of frames (time steps) in the sequence.
            dim_input (int): The input dimension (patch embedding dimension).
            num_heads (int, optional): The number of attention heads. Defaults to 8.
            dim_head (int, optional): The dimension of each attention head. Defaults to 64.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """

        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim_input)

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scaling_factor = dim_head**-0.5  # Scale factor for dot product attention
        self.norm = nn.LayerNorm(dim_input)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Jointly calculate Query, Key, and Value
        self.to_qkv = nn.Linear(dim_input, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim_input), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        # Generate the fixed block-causal mask based on current global settings
        self.bias = generate_block_causal_mask_matrix(num_patches, num_frames).to("cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-head self-attention mechanism with a fixed block-causal mask.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim_input) after processing through the attention mechanism.
        """
        B, T, C = x.size()  # T is the sequence length i.e. (num_frames * num_patches)
        x = self.norm(x)

        # Split and rearrange for multi-head attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        # Calculate scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scaling_factor

        # Apply block-causal mask: set 'future' scores to -infinity so softmax makes them 0 (i.e. not attended to)
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Combine head outputs
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    A Transformer consisting of multiple layers of block-causal multi-head self-attention and feedforward networks.
    Expects a fixed length input sequence of patches across multiple frames, with the attention mechanism using
    a fixed block-causal mask to ensure proper temporal modeling.
    """

    def __init__(
        self,
        num_patches: int,
        num_frames: int,
        dim_input: int,
        depth: int,
        num_heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            num_patches (int): The number of patches per frame.
            num_frames (int): The number of frames (time steps) in the sequence.
            dim_input (int): The input dimension (patch embedding dimension).
            depth (int): The number of Transformer layers (attention + feedforward).
            num_heads (int): The number of attention heads at each layer.
            dim_head (int): The dimension of each attention head.
            mlp_dim (int): The hidden dimension of the feedforward network.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim_input)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            num_patches=num_patches,
                            num_frames=num_frames,
                            dim_input=dim_input,
                            num_heads=num_heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(dim_input, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x  # Self-attention + Residual
            x = ff(x) + x  # MLP + Residual
        return self.norm(x)


class TransitionModel(nn.Module):
    """
    This is a Transition Model that models the dynamics of the environment in the latent space.
    It adopts a ViT architecture with a custom fixed block-causal attention mask to ensure proper temporal modeling for a
    sequence of patches across multiple frames (which means it must be initialized to process sequences of a fixed length).
    This input sequence is essentially a history of observation latents across multiple frames.
    Action and proprioceptive information is assumed to already be incorporated, either into the patch tokens themselves
    (e.g. via concatenation) or by passing them in as their own patch tokens.
    """

    def __init__(
        self,
        *,
        num_patches: int,
        num_frames: int,
        dim_input: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        """
        Initializes the Transition Model.

        Args:
            num_patches (int): The number of patches per frame.
            num_frames (int): The number of frames (time steps) in the sequence.
            dim_input (int): The dimension of the input embeddings.
            depth (int): The number of Transformer layers.
            num_heads (int): The number of attention heads.
            mlp_dim (int): The hidden dimension of the feedforward network.
            dim_head (int, optional): The dimension of each attention head. Defaults to 64.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            emb_dropout (float, optional): The dropout probability for the position embeddings. Defaults to 0.0.
        """

        super().__init__()

        # Learnable position embeddings for every patch across every frame in the window
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim_input))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            num_patches=num_patches,
            num_frames=num_frames,
            dim_input=dim_input,
            depth=depth,
            num_heads=num_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transition Model.


        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames * num_patches, dim_input),
                representing the sequence of patch embeddings across multiple frames.

        Returns:
            torch.Tensor: The full processed sequence after modeling temporal dynamics,
                with the same shape (batch_size, num_frames * num_patches, dim_input).
        """
        # x shape: (batch, num_frames * num_patches, embedding_dim)
        b, n, _ = x.shape

        # Add temporal and spatial position information
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        # Process through the causal transformer
        x = self.transformer(x)

        # Return the full processed sequence
        return x
