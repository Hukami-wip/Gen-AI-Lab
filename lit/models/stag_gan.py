import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim import Adam

from models.stag.stag_gan import StagDiscriminator, StagGenerator


class StagGAN(LightningModule):
    """
    StyleGAN2-inspired GAN for pre-training sentence embedding generation
    Supports both one-shot and sequence generation modes
    """

    def __init__(
        self,
        # Generator params
        model_dim: int = 1024,  # SONAR embedding dimension
        state_dim: int = 2048,
        num_heads: int = 32,
        num_layers: int = 1,
        z_dim: int = 512,
        w_dim: int = 512,
        mapping_layers: int = 8,
        initial_state_type: str = "learnable",
        # Discriminator params
        disc_hidden_dims: list = None,
        use_spectral_norm: bool = True,
        minibatch_stddev: bool = True,
        disc_input_noise: float = 0.01,  # Noise added to discriminator inputs
        # Training params
        g_lr: float = 1e-4,
        d_lr: float = 1e-4,
        beta1: float = 0.0,
        beta2: float = 0.99,
        r1_gamma: float = 1.0,  # R1 regularization weight
        n_critic: int = 1,  # Train discriminator n times per generator step
    ):
        super().__init__()
        self.save_hyperparameters()

        if disc_hidden_dims is None:
            disc_hidden_dims = [512, 256, 128]

        # Initialize generator and discriminator
        self.generator = StagGenerator(
            model_dim=model_dim,
            state_dim=state_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            z_dim=z_dim,
            w_dim=w_dim,
            mapping_layers=mapping_layers,
            initial_state_type=initial_state_type,
        )

        self.discriminator = StagDiscriminator(
            model_dim=model_dim,
            hidden_dims=disc_hidden_dims,
            use_spectral_norm=use_spectral_norm,
            minibatch_stddev=minibatch_stddev,
        )

        # For tracking training metrics
        self.automatic_optimization = False

        # For logging penalties
        self.last_r1_penalty = 0.0

    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator."""
        g_mapping_params = self.generator.mapping.parameters()
        g_synthesis_params = [
            p for name, p in self.generator.named_parameters() if "mapping." not in name
        ]

        g_optimizer = Adam(
            [
                {
                    "params": g_synthesis_params,
                    "lr": self.hparams.g_lr,
                    "betas": (self.hparams.beta1, self.hparams.beta2),
                },
                {
                    "params": g_mapping_params,
                    "lr": self.hparams.g_lr * self.generator.mapping.lr_mul,
                    "betas": (self.hparams.beta1, self.hparams.beta2),
                },
            ]
        )

        d_optimizer = Adam(
            self.discriminator.parameters(),
            lr=self.hparams.d_lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )

        return [g_optimizer, d_optimizer]

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        sentence_embeddings, _ = batch
        real_embeddings = sentence_embeddings.view(-1, self.hparams.model_dim)
        batch_size = real_embeddings.shape[0]

        d_opt.zero_grad()

        # Generate fake samples
        z = torch.randn(batch_size, self.hparams.z_dim, device=self.device)
        fake_embeddings, _ = self.generator(z)

        # Add noise to discriminator inputs
        noise_scale = max(self.hparams.disc_input_noise, 0.02)
        if noise_scale > 0:
            real_embeddings_noisy = real_embeddings + (
                torch.randn_like(real_embeddings) * noise_scale
            )
            fake_embeddings_noisy = fake_embeddings.detach() + (
                torch.randn_like(fake_embeddings) * noise_scale
            )
        else:
            real_embeddings_noisy = real_embeddings
            fake_embeddings_noisy = fake_embeddings.detach()

        # Discriminator forward pass on embeddings signals
        real_pred = self.discriminator(real_embeddings_noisy)
        fake_pred = self.discriminator(fake_embeddings_noisy)

        # Adversarial loss
        d_loss_real = F.softplus(-real_pred).mean()
        d_loss_fake = F.softplus(fake_pred).mean()
        d_loss = d_loss_real + d_loss_fake

        # R1 Regularization
        if self.hparams.r1_gamma > 0 and batch_idx % 4 == 0:
            real_embeddings.requires_grad = True
            grad_real = torch.autograd.grad(
                outputs=self.discriminator(real_embeddings).sum(),
                inputs=real_embeddings,
                create_graph=True,
            )[0]
            r1_penalty = (
                grad_real.pow(2).view(real_embeddings.shape[0], -1).sum(1).mean()
            )
            r1_penalty = self.hparams.r1_gamma / 2 * r1_penalty

            d_loss += r1_penalty
            self.last_r1_penalty = r1_penalty.item()

        self.manual_backward(d_loss)
        d_opt.step()

        # Train Generator
        if batch_idx % self.hparams.n_critic == 0:
            g_opt.zero_grad()
            z = torch.randn(batch_size, self.hparams.z_dim, device=self.device)
            # Generate fake samples
            fake_embeddings, w = self.generator(z)

            fake_pred = self.discriminator(fake_embeddings)
            g_loss = F.softplus(-fake_pred).mean()

            self.manual_backward(g_loss)
            g_opt.step()

            self.log("g_loss", g_loss, prog_bar=True)

        # Log metrics
        self.log("d_loss", d_loss, prog_bar=True)
        self.log("r1_penalty", self.last_r1_penalty, prog_bar=False)
        self.log("real_score", real_pred.mean(), prog_bar=False)
        self.log("fake_score", fake_pred.mean(), prog_bar=False)

        if torch.cuda.is_available():
            self.log(
                "memory_allocated_gb",
                torch.cuda.memory_allocated() / 1e9,
                prog_bar=False,
            )
            self.log(
                "memory_reserved_gb",
                torch.cuda.memory_reserved() / 1e9,
                prog_bar=False,
            )

    def validation_step(self, batch, batch_idx):
        """Validation step - calculates scores on the validation set."""
        sentence_embeddings, _ = batch
        real_embeddings = sentence_embeddings.view(-1, self.hparams.model_dim)
        batch_size = real_embeddings.shape[0]

        with torch.no_grad():
            z = torch.randn(batch_size, self.hparams.z_dim, device=self.device)
            fake_embeddings, _ = self.generator(z)

            real_pred = self.discriminator(real_embeddings)
            fake_pred = self.discriminator(fake_embeddings)

            self.log("val_real_score", real_pred.mean())
            self.log("val_fake_score", fake_pred.mean())

    def sample_embeddings(self, num_samples=8):
        """Generate sample embeddings for inspection"""
        self.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, self.hparams.z_dim, device=self.device)
            samples = self.generator(z)

        self.train()
        return samples

    def get_pretrained_generator(self):
        """
        Return the generator for reuse in sequence training
        This allows the pre-trained model to be used for autoregressive training
        """
        return self.generator

    def convert_to_stagcm(self):
        """
        Convert the pre-trained generator to a StagCM for sequence training
        Returns the Stag backbone with learned initial state (BOS)
        """
        # Extract the Stag backbone (which contains the learned initial state)
        stag_backbone = self.generator.stag

        # Return components for reuse
        return {
            "stag_model": stag_backbone,  # Contains learned initial_state
            "mapping_network": self.generator.mapping,
            "style_projection": self.generator.style_projection,
        }
