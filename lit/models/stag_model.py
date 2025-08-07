import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from muon import Muon

from models.stag import Stag


class StagLitModel(LightningModule):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        num_heads: int,
        num_layers: int,
        output_dim: int,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Enable manual optimization for multiple optimizers
        self.automatic_optimization = False

        self.embedding = nn.Linear(1, self.hparams.model_dim)

        self.model = Stag(
            model_dim=self.hparams.model_dim,
            state_dim=self.hparams.state_dim,
            num_heads=self.hparams.num_heads,
            num_layers=self.hparams.num_layers,
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                self.hparams.model_dim + self.hparams.output_dim,
                # self.hparams.model_dim * 2,
                self.hparams.model_dim,
            ),
            # nn.ReLU(),
            # nn.Linear(self.hparams.model_dim * 2, self.hparams.model_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.model_dim, self.hparams.output_dim),
        )
        self.pos_embedding = nn.Embedding(
            self.hparams.output_dim, self.hparams.model_dim
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        context_vector = self.model(x)
        # One-hot position encoding instead of learned embeddings
        positions = (
            torch.eye(self.hparams.output_dim, device=x.device)
            .unsqueeze(0)
            .expand(x.shape[0], -1, -1)
        )  # (batch, seq_len, seq_len)

        # Concatenate context with one-hot positions
        context_expanded = context_vector.unsqueeze(1).expand(
            -1, self.hparams.output_dim, -1
        )
        decoder_input = torch.cat([context_expanded, positions], dim=-1)

        x = self.decoder(decoder_input)
        return x

    def training_step(self, batch, batch_idx):
        # Manual optimization
        optimizers = self.optimizers()

        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.output_dim), targets.view(-1)
        )

        # Manual backward pass
        self.manual_backward(loss)

        # Manual gradient clipping (if you had gradient_clip_val in trainer)
        self.clip_gradients(
            optimizers[0], gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        if len(optimizers) > 1:
            self.clip_gradients(
                optimizers[1], gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )

        # Step each optimizer
        for opt in optimizers:
            opt.step()
            opt.zero_grad()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.output_dim), targets.view(-1)
        )

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=-1)
        acc = (preds == targets).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.output_dim), targets.view(-1)
        )

        preds = torch.argmax(outputs, dim=-1)
        acc = (preds == targets).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        # Separate parameters for muon (2D weights) and standard optimizer (others)
        muon_params = []
        standard_params = []

        for name, param in self.named_parameters():
            if (
                param.ndim >= 2
                and "embedding" not in name.lower()
                and "decoder" not in name.lower()
                and "model." in name
            ):  # STAG model weights
                muon_params.append(param)
            else:
                standard_params.append(param)

        # Create separate optimizers
        optimizers = []

        if muon_params:
            muon_optimizer = Muon(
                muon_params,
                lr=self.hparams.lr * 20,  # Higher LR for muon
                momentum=0.95,
            )
            optimizers.append(muon_optimizer)

        if standard_params:
            adam_optimizer = torch.optim.AdamW(
                standard_params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
            optimizers.append(adam_optimizer)

        return optimizers
