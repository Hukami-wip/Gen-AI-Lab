import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from muon import SingleDeviceMuonWithAuxAdam

from models.stag import Stag


class StagCM(LightningModule):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        num_heads: int,
        num_layers: int,
        # Pre-training options
        use_pretrained_gan: bool = False,
        gan_checkpoint_path: str = None,
        freeze_stag_backbone: bool = False,
        # Optimization
        muon_lr: float = 2e-3,
        scalar_lr: float = 1e-4,
        weight_decay_adam: float = 1e-4,
        weight_decay_muon: float = 0.0,
        momentum: float = 0.95,
        betas=(0.9, 0.95),
        eps: float = 1e-10,
        scheduler_cfg: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize or load pre-trained components
        if use_pretrained_gan and gan_checkpoint_path:
            self._load_from_pretrained_gan(gan_checkpoint_path)
        else:
            # Initialize fresh model
            self.model = Stag(
                model_dim=self.hparams.model_dim,
                state_dim=self.hparams.state_dim,
                num_heads=self.hparams.num_heads,
                num_layers=self.hparams.num_layers,
            )

        self.criterion = nn.MSELoss()
        self.scheduler_cfg = scheduler_cfg or {}

        # Freeze backbone if requested
        if freeze_stag_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_from_pretrained_gan(self, checkpoint_path):
        """Load pre-trained components from StagGAN checkpoint"""
        print(f"Loading pre-trained GAN from: {checkpoint_path}")

        # Import here to avoid circular imports
        from .stag_gan import StagGAN

        # Load the GAN checkpoint
        gan_model = StagGAN.load_from_checkpoint(checkpoint_path)
        pretrained_components = gan_model.convert_to_stagcm()

        # Extract the Stag model (which includes learned initial state)
        self.model = pretrained_components["stag_model"]

        # Store additional pre-trained components (optional for advanced usage)
        self.mapping_network = pretrained_components["mapping_network"]
        self.style_projection = pretrained_components["style_projection"]

        print("Successfully loaded pre-trained GAN components!")

    def forward(self, x):
        """
        x: (batch, seq_len, model_dim)
        """
        return self.model(x)

    def _shared_step(self, batch):
        inputs = batch[:, :-1, :]
        targets = batch[:, 1:, :]

        predictions = self.forward(inputs)

        loss = self.criterion(predictions, targets)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss)
        return loss

    def generate_with_initial_state(self, batch_size, max_length=10, device=None):
        """
        Generate sequences starting with the learned initial state
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            # Create dummy input to trigger the initial state
            # The actual values don't matter since we only want the first output
            dummy_input = torch.zeros(
                batch_size, 1, self.hparams.model_dim, device=device
            )

            # Get first output from initial state
            first_output = self.forward(dummy_input)[:, 0, :]  # (batch_size, model_dim)
            sequence = [first_output]

            for _ in range(max_length - 1):
                # Convert to sequence format for model
                input_seq = torch.stack(sequence, dim=1)  # (batch, seq_len, model_dim)

                # Get prediction for next token
                output = self.forward(input_seq)
                next_token = output[:, -1, :]  # Last prediction

                sequence.append(next_token)

            result = torch.stack(sequence, dim=1)

        self.train()
        return result

    def generate_from_style(self, z, max_length=10):
        """
        Generate sequences using pre-trained style components (if available)
        Only works if model was loaded from pre-trained GAN
        """
        if not hasattr(self, "mapping_network"):
            raise ValueError("Style generation requires pre-trained GAN components")

        batch_size = z.shape[0]
        device = z.device

        self.eval()
        with torch.no_grad():
            # Map noise to style
            w = self.mapping_network(z)

            # Generate initial embedding from style
            initial_embedding = self.style_projection(w)
            sequence = [initial_embedding]

            for _ in range(max_length - 1):
                input_seq = torch.stack(sequence, dim=1)
                output = self.forward(input_seq)
                next_token = output[:, -1, :]
                sequence.append(next_token)

            result = torch.stack(sequence, dim=1)

        self.train()
        return result

    def configure_optimizers(self):
        muon_params = []
        muon_params_names = []
        adam_params = []
        adam_params_names = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if p.ndim >= 2 and "W_I" not in name and "W_O" not in name:
                muon_params.append(p)  # Muon
                muon_params_names.append(name)
            else:
                adam_params.append(p)  # Adam (bias, ln weights)
                adam_params_names.append(name)

        param_groups = []

        if adam_params:
            param_groups.append(
                dict(
                    params=adam_params,
                    lr=self.hparams.scalar_lr,
                    betas=self.hparams.betas,
                    eps=self.hparams.eps,
                    weight_decay=self.hparams.weight_decay_adam,
                    use_muon=False,
                )
            )

        if muon_params:
            param_groups.append(
                dict(
                    params=muon_params,
                    lr=self.hparams.muon_lr,
                    momentum=self.hparams.momentum,
                    weight_decay=self.hparams.weight_decay_muon,
                    use_muon=True,
                )
            )

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        return optimizer
