import torch
import torch.nn as nn
import torch.nn.functional as nf

from .conv import MambaConv
from .input import MambaInput
from .state_space_model import SSM


class MambaBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        conv_dim: int,
        expand_factor: int = 1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.conv_dim = conv_dim
        self.expand_factor = expand_factor

        self.input_block = MambaInput(model_dim, expand_factor)
        self.conv_block = MambaConv(self.input_block.model_dim, conv_dim)
        self.ssm_block = SSM(self.input_block.model_dim, state_dim)

        self.output_proj = nn.Linear(self.input_block.model_dim, self.model_dim)

        self.norm = nn.LayerNorm(self.model_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, seq_len, model_dim)
        """
        residual = x

        x = self.norm(x)

        x_main, z_gate = self.input_block(x)
        x_main = self.conv_block(x_main)
        y_ssm = self.ssm_block(x_main)

        gated_output = y_ssm * nf.silu(z_gate)

        x = self.output_proj(gated_output)

        return x + residual
