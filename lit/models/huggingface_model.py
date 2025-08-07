from lightning.pytorch import LightningModule
from torch.optim import AdamW


class HfModel(LightningModule):
    """
    A LightningModule for a Hugging Face model.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        lr: float = 1e-4,
        num_classes: int = 10,
    ):
        super().__init__()
        from transformers import AutoModelForImageClassification

        self.save_hyperparameters()
        self.lr = lr
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("val_loss", outputs.loss)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("test_loss", outputs.loss)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
