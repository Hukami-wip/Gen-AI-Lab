import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class SimpleLCM(LightningModule):
    def __init__(
        self,
        input_dim: int = 1024,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        lr: float = 1e-4,
        min_context_size: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = input_dim
        self.min_context_size = min_context_size

        # Use only encoder for next-sentence prediction
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
        self.criterion = nn.MSELoss()

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        # For next-sentence prediction, we use the transformer encoder
        output = self.transformer(src)
        return output

    def _prepare_target_mask(self, batch_size, seq_len, device):
        """Create target mask for next-sentence prediction"""
        # Create mask where True positions are valid targets
        # We don't predict the first position (min_context_size positions)
        target_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        target_mask[:, : self.min_context_size] = False
        return target_mask

    def training_step(self, batch, batch_idx):
        # batch shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = batch.shape
        device = batch.device

        # Forward pass
        output = self.forward(batch)

        # Prepare targets: shift input by 1 position
        # Input: [s1, s2, s3, s4, s5] -> Target: [s2, s3, s4, s5]
        target_seqs = batch[:, 1:].contiguous()

        # Prepare predictions: remove last position
        # Output: [ŝ1, ŝ2, ŝ3, ŝ4, ŝ5] -> Predictions: [ŝ1, ŝ2, ŝ3, ŝ4]
        predicted_seqs = output[:, :-1].contiguous()

        # Create target mask
        target_mask = self._prepare_target_mask(batch_size, seq_len - 1, device)

        # Flatten and apply mask
        flattened_predictions = predicted_seqs.view(-1, input_dim)[target_mask.view(-1)]
        flattened_target = target_seqs.view(-1, input_dim)[target_mask.view(-1)]

        # Compute loss only on masked positions
        loss = self.criterion(flattened_predictions, flattened_target)

        # Log metrics
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("num_target_elements", target_mask.sum().item(), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size, seq_len, input_dim = batch.shape
        device = batch.device

        # Forward pass
        output = self.forward(batch)

        # Prepare targets and predictions (same as training)
        target_seqs = batch[:, 1:].contiguous()
        predicted_seqs = output[:, :-1].contiguous()

        # Create target mask
        target_mask = self._prepare_target_mask(batch_size, seq_len - 1, device)

        # Flatten and apply mask
        flattened_predictions = predicted_seqs.view(-1, input_dim)[target_mask.view(-1)]
        flattened_target = target_seqs.view(-1, input_dim)[target_mask.view(-1)]

        # Compute loss
        loss = self.criterion(flattened_predictions, flattened_target)

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_num_target_elements", target_mask.sum().item(), on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
