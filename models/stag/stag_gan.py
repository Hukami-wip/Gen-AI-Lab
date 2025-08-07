import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .stag import Stag


class MappingNetwork(nn.Module):
    """z → w mapping network (StyleGAN2 inspired)"""

    def __init__(self, z_dim=512, w_dim=512, num_layers=8, lr_mul=0.01):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            out_dim = w_dim

            layers.extend([nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2)])

        self.mapping = nn.Sequential(*layers)
        self.lr_mul = lr_mul

        # Initialize with smaller weights
        for layer in self.mapping:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.02 * lr_mul)
                nn.init.zeros_(layer.bias)

    def forward(self, z):
        """
        z: (batch_size, z_dim)
        Returns: w (batch_size, w_dim)
        """
        return self.mapping(z)


class Style(nn.Module):
    """Style block.
    A simple linear layer to project the latent code w to a style vector.
    """

    def __init__(self, w_dim, style_dim):
        super().__init__()
        # Project w to get both scale (gamma) and bias (beta)
        self.linear = nn.Linear(w_dim, style_dim * 2)

    def forward(self, w):
        # style will have shape (batch_size, style_dim * 2)
        style = self.linear(w)
        # Split into scale and bias
        scale, bias = style.chunk(2, dim=1)
        return scale, bias


class StagGenerator(nn.Module):
    """
    Generator that uses STAG backbone for one-shot embedding generation
    Style vector is modulated at each layer of the STAG model.
    """

    def __init__(
        self,
        model_dim=1024,  # SONAR embedding dim
        state_dim=2048,
        num_heads=32,
        num_layers=1,
        z_dim=512,
        w_dim=512,
        mapping_layers=8,
        initial_state_type="learnable",
    ):
        super().__init__()

        self.model_dim = model_dim
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = state_dim // num_heads

        # Mapping network z → w
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers)

        # # Topic projection to shape it for the STAG initial state
        # self.topic_projection = nn.Linear(model_dim, state_dim * num_layers)

        # Per-layer style projections
        self.style_projections = nn.ModuleList(
            [Style(w_dim, state_dim) for _ in range(num_layers)]
        )

        # STAG backbone
        self.stag = Stag(
            model_dim=model_dim,
            state_dim=state_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            initial_state_type=initial_state_type,
        )

    def forward(self, z):
        """
        One-shot generation: z + topic → single sentence embedding via STAG
        z: (batch_size, z_dim)
        topic_embedding: (batch_size, model_dim)
        """
        batch_size = z.shape[0]

        # z → w mapping
        w = self.mapping(z)

        # Generate per-layer styles
        styles = [proj(w) for proj in self.style_projections]

        # A dummy input for STAG
        dummy_input = torch.zeros(batch_size, 1, self.model_dim, device=z.device)

        # Feed styles and the new initial state to STAG
        output = self.stag(dummy_input, styles=styles)

        # Return single embedding and w
        return output.squeeze(1), w

    def generate_embeddings(self, num_samples=8, device=None):
        """Generate sample embeddings for inspection"""
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.mapping.mapping[0].in_features, device=device)
        return self.forward(z)[0]


class StagDiscriminator(nn.Module):
    """
    Discriminator for sentence embeddings with spectral norm and minibatch stddev
    """

    def __init__(
        self,
        model_dim=1024,
        hidden_dims=[512, 256, 128],
        use_spectral_norm=True,
        minibatch_stddev=True,
    ):
        super().__init__()

        self.minibatch_stddev = minibatch_stddev

        layers = []
        in_dim = model_dim

        # Add minibatch stddev feature if enabled
        if minibatch_stddev:
            in_dim += 1

        # Build discriminator layers
        for hidden_dim in hidden_dims:
            linear = nn.Linear(in_dim, hidden_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear)

            layers.extend([linear, nn.LeakyReLU(0.2), nn.Dropout(0.1)])
            in_dim = hidden_dim

        # Final output layer
        final_linear = nn.Linear(in_dim, 1)
        if use_spectral_norm:
            final_linear = spectral_norm(final_linear)
        layers.append(final_linear)

        self.discriminator = nn.Sequential(*layers)

    def minibatch_standard_deviation(self, x):
        """
        Compute minibatch standard deviation feature
        x: (batch_size, model_dim)
        Returns: (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Compute std across batch dimension
        y = x - x.mean(dim=0, keepdim=True)  # Center
        y = (y**2).mean(dim=0)  # Variance
        y = (y + 1e-8).sqrt()  # Std
        y = y.mean()  # Average std across features

        # Replicate for each sample in batch
        y = y.repeat(batch_size, 1)
        return y

    def forward(self, x):
        """
        x: (batch_size, model_dim) - sentence embeddings
        Returns: (batch_size, 1) - real/fake logits
        """
        if self.minibatch_stddev:
            stddev_feature = self.minibatch_standard_deviation(x)
            x = torch.cat([x, stddev_feature], dim=1)

        return self.discriminator(x)
