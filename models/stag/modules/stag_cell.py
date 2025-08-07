import torch.nn as nn
import torch.nn.functional as nf


class StagCell(nn.Module):
    def __init__(self, state_dim, num_heads):
        super().__init__()
        self.state_dim = state_dim  # D
        self.num_heads = num_heads  # N
        assert self.state_dim % num_heads == 0, (
            "state_dim must be divisible by num_heads"
        )
        self.head_dim = state_dim // num_heads  # S

        # --- g(h,x) projections: Gated state evolution ---
        self.W_uQ = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.W_uK = nn.Linear(self.state_dim, self.state_dim, bias=False)

        # --- h(h,x) projections: Gated input injection ---
        self.W_vQ = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.W_vK = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.W_vV = nn.Linear(self.state_dim, self.state_dim, bias=False)

        # Layer Normalization for the hidden state
        self.norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, h):
        """
        x: current input token -> (batch, state_dim)
        h: previous state -> (batch, num_heads, head_dim)
        """
        batch_size = h.shape[0]

        # Normalize the hidden state before it's used for gating
        h_flat = h.view(batch_size, self.state_dim)

        # --- 1. Calculate u(h,x): The gated previous state ---
        u_Q = self.W_uQ(h_flat).view(batch_size, self.num_heads, self.head_dim)
        u_K = self.W_uK(x).view(batch_size, self.num_heads, self.head_dim)

        # per-head gating
        u_scores = (u_Q * u_K).sum(dim=-1, keepdim=True) / (self.head_dim**0.5)
        u_attn_gate = nf.sigmoid(u_scores)

        # softmax gate
        # u_attn_gate = nf.softmax(u_Q * u_K / (self.head_dim**0.5), dim=-1)

        # sigmoid gate
        # u_attn_gate = nf.sigmoid(u_Q * u_K)

        u_out = u_attn_gate * h

        # --- 2. Calculate v(h,x): The gated new input ---
        v_Q = self.W_vQ(h_flat).view(batch_size, self.num_heads, self.head_dim)
        v_K = self.W_vK(x).view(batch_size, self.num_heads, self.head_dim)
        v_V = self.W_vV(x).view(batch_size, self.num_heads, self.head_dim)

        # per-head gating
        # v_scores = (v_Q * v_K).sum(dim=-1, keepdim=True) / (self.head_dim**0.5)
        # v_attn_gate = nf.sigmoid(v_scores)
        # v_out = v_attn_gate * v_V  # Broadcast the gate

        # softmax gate
        v_attn_gate = nf.softmax((v_Q * v_K) / (self.head_dim**0.5), dim=-1)

        # sigmoid gate
        # v_attn_gate = nf.sigmoid(v_Q * v_K)

        v_out = v_attn_gate * v_V

        # --- 3. Combine heads and produce final state ---
        h_next = u_out + v_out

        h_next_norm = self.norm(h_next)

        return h_next_norm
