import torch
import torch.nn as nn

from .modules.mamba_block import MambaBlock


class MambaViT(nn.Module):
    """
    MambaViT architecture - pure PyTorch model without training logic
    """

    def __init__(
        self,
        model_dim: int,
        n_layers: int,
        n_classes: int,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand_factor: int = 2,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.state_dim = state_dim
        self.conv_dim = conv_dim
        self.expand_factor = expand_factor

        self.embedding = nn.Linear(1, self.model_dim)

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    model_dim=model_dim,
                    state_dim=state_dim,
                    conv_dim=conv_dim,
                    expand_factor=expand_factor,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.model_dim)
        self.head = nn.Linear(self.model_dim, self.n_classes)

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, seq_len, 1)
        """
        x = x.unsqueeze(-1)
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x[:, -1, :])

        x = self.head(x)

        return x
