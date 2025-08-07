import torch.nn as nn


class MambaInput(nn.Module):
    """
    Project the input from `input_dim` to `2 * `model_dim` and split it into the main data path `x` and the gate `z`.
    """

    def __init__(self, input_dim: int, expand_factor: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = int(expand_factor * input_dim)

        # Linear layer for the main data path
        # self.x_proj = nn.Linear(input_dim, self.model_dim, bias=False)

        # Linear layer for the gate path
        # self.z_proj = nn.Linear(input_dim, self.model_dim, bias=False)

        # A single projection layer for both main and gate paths
        self.in_proj = nn.Linear(self.input_dim, 2 * self.model_dim, bias=False)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """

        x_proj = self.in_proj(x)

        x_main, z_gate = x_proj.chunk(2, dim=-1)

        return x_main, z_gate
