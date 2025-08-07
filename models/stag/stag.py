import torch
import torch.nn as nn

from .modules.stag_cell import StagCell


class Stag(nn.Module):
    """
    STAG Backbone model.
    This module processes a sequence through multiple STAG layers.
    """

    def __init__(
        self,
        model_dim,
        state_dim,
        num_heads,
        num_layers,
        ffn_dim=None,
        dropout=0.1,
        initial_state_type="learnable",
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        assert state_dim % num_heads == 0, "state_dim must be divisible by num_heads"
        self.state_dim = state_dim
        self.head_dim = state_dim // num_heads
        self.num_layers = num_layers
        self.initial_state_type = initial_state_type
        assert self.initial_state_type in [
            "learnable",
            "zero",
            "random",
        ], "initial_state_type must be one of 'learnable', 'zero', or 'random'"

        # Default FFN dimension to 2x state_dim
        if ffn_dim is None:
            ffn_dim = 2 * state_dim

        self.cells = nn.ModuleList(
            [StagCell(self.state_dim, num_heads) for _ in range(num_layers)]
        )

        # FFN blocks for each layer
        self.ffn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.state_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, self.state_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.W_I = nn.Linear(self.model_dim, self.state_dim, bias=False)
        self.W_O = nn.Linear(self.state_dim, self.model_dim, bias=False)

        self.initial_state = None
        if self.initial_state_type == "learnable":
            self.initial_state = nn.Parameter(
                torch.randn(num_layers, num_heads, self.head_dim)
            )

    def forward(self, x_seq, styles=None, initial_state=None):
        """
        x_seq: (batch, seq_len, model_dim)
        styles: A list of style vectors, one for each layer.
        initial_state: An optional external initial state.
        """
        batch_size, seq_len, _ = x_seq.shape

        if initial_state is not None:
            h = initial_state
        elif self.initial_state_type == "learnable":
            h = self.initial_state.unsqueeze(1).expand(-1, batch_size, -1, -1)
        elif self.initial_state_type == "zero":
            h = torch.zeros(
                self.num_layers,
                batch_size,
                self.num_heads,
                self.head_dim,
                device=x_seq.device,
            )
        else:  # random
            h = torch.randn(
                self.num_layers,
                batch_size,
                self.num_heads,
                self.head_dim,
                device=x_seq.device,
            )

        outputs = []
        for t in range(seq_len):
            x_t = self.W_I(x_seq[:, t, :])

            h_layers = []

            layer_input = x_t
            for i, (cell, ffn) in enumerate(zip(self.cells, self.ffn_blocks)):
                h_t = cell(layer_input, h[i])
                h_layers.append(h_t)
                layer_output = h_t.view(batch_size, self.state_dim)

                # Apply residual connection for STAG cell
                layer_output = layer_output + layer_input

                # Apply FFN with residual connection
                ffn_output = ffn(layer_output) + layer_output

                # Apply style modulation if styles are provided
                if styles is not None:
                    scale, bias = styles[i]
                    layer_input = ffn_output * (1 + scale) + bias
                else:
                    layer_input = ffn_output

            h = torch.stack(h_layers)
            outputs.append(layer_input)

        outputs = torch.stack(outputs, dim=1)
        return self.W_O(outputs)
