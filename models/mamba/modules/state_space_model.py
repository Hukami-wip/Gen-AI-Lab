import math

import torch
import torch.nn as nn
import torch.nn.functional as nf

from .selective_scan import selective_scan


class SSM(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        dt_min=0.001,
        dt_max=0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim

        # Projections for selective parameters, delta, B, and C
        self.delta_proj = nn.Linear(self.model_dim, self.model_dim, bias=True)

        self.B_proj = nn.Linear(self.model_dim, self.state_dim, bias=False)
        self.C_proj = nn.Linear(self.model_dim, self.state_dim, bias=False)

        # Matrix A is not data-dependent
        A = torch.arange(1, self.state_dim + 1, dtype=torch.float32).repeat(
            self.model_dim, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Skip connection D.
        self.D = nn.Parameter(torch.ones(self.model_dim))
        self.D._no_weight_decay = True

        nn.init.zeros_(self.delta_proj.weight)
        dt_init = torch.exp(
            torch.rand(self.model_dim) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))

        with torch.no_grad():
            self.delta_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, seq_len, model_dim)
        """

        # Project the input
        delta = nf.softplus(self.delta_proj(x))
        B = self.B_proj(x)
        C = self.C_proj(x)

        A = -torch.exp(self.A_log.float())

        D = self.D.float()

        y = selective_scan(x, delta, A, B, C, D)

        return y
