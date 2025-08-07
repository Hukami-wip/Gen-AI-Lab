import torch
import torch.nn as nn


class MambaConv(nn.Module):
    """
    Applies a 1D causal depthwise convolution.
    """

    def __init__(self, model_dim: int, conv_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.conv_dim = conv_dim

        self.conv = nn.Conv1d(
            in_channels=self.model_dim,
            out_channels=self.model_dim,
            bias=True,
            kernel_size=self.conv_dim,
            groups=self.model_dim,
            padding=self.conv_dim - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape

        x_permuted = x.permute(0, 2, 1)

        x_conv = self.conv(x_permuted)[:, :, :L]

        x_conv_permuted = x_conv.permute(0, 2, 1)

        return x_conv_permuted
