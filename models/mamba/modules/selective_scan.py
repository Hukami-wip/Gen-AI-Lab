import torch
from einops import einsum


def selective_scan(x, delta, A, B, C, D):
    """
    Performs the selective scan operation.
    """
    B_batch, L_len, model_dim = x.shape
    N_state = A.shape[1]

    # 1. Discretize A and B
    # A_bar = exp(delta * A)
    # B_bar = (exp(delta * A) - 1) / A * B

    # Discretize A: (B, L, model_dim, N)
    delta_A = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))

    # Discretize B: B_bar = delta * B
    # B_bar is an outer product of delta and B, for each item in batch and sequence.
    # (B, L, D, N)
    delta_B = einsum(delta, B, "b l d, b l n -> b l d n")

    # The state update term B_bar * x
    # (B, L, D, N) * (B, L, D, 1) -> (B, L, D, N)
    delta_B_x = delta_B * x.unsqueeze(-1)

    # 2. Perform the scan operation (recurrent loop)
    # Initialize the hidden state h to zeros.
    h = torch.zeros(B_batch, model_dim, N_state, device=x.device)

    # A list to store the output y at each timestep.
    ys = []

    for i in range(L_len):
        # Get the parameters for the current timestep
        h = delta_A[:, i] * h + delta_B_x[:, i]
        ys.append(h)

    # Stack the outputs into a single tensor
    # h_stacked is (L, B, model_dim, N)
    h_stacked = torch.stack(ys, dim=1)  # (B, L, model_dim, N)

    # 3. Compute the final output y
    # y = C * h
    y = einsum(h_stacked, C, "b l d n, b l n -> b l d")

    # 4. Add the skip connection D * x
    y = y + x * D.unsqueeze(0)

    return y
