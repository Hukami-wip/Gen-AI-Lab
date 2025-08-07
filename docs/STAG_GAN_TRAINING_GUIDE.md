# StagGAN Pre-training Guide

This guide explains how to use the configuration files for pre-training StagGAN and transferring to sequence tasks.

## 📁 Configuration Files Overview

### Data Configs (`configs/data/`)

- **`sonar_gan.yaml`** - Individual embeddings for GAN pre-training
- **`sonar_sequences.yaml`** - Sequences for autoregressive training

### Model Configs (`configs/model/`)

- **`stag_gan.yaml`** - StagGAN model configuration
- **`stag_cm_pretrained.yaml`** - StagCM with pre-trained components

### Trainer Configs (`configs/trainer/`)

- **`train_stag_gan.yaml`** - GAN training configuration
- **`train_stag_cm_pretrained.yaml`** - Sequence training configuration

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

```bash
# Run the complete pipeline: GAN pre-training → sequence training → evaluation
python scripts/train_stag_gan_pipeline.py --phase all
```

### Option 2: Step-by-Step Training

#### Step 1: Pre-train StagGAN

```bash
python lit_cli.py fit \
    --config configs/trainer/train_stag_gan.yaml \
    --model configs/model/stag_gan.yaml \
    --data configs/data/sonar_gan.yaml
```

#### Step 2: Train StagCM with pre-trained components

```bash
# Update the checkpoint path in configs/model/stag_cm_pretrained.yaml first!
python lit_cli.py fit \
    --config configs/trainer/train_stag_cm_pretrained.yaml \
    --model configs/model/stag_cm_pretrained.yaml \
    --data configs/data/sonar_sequences.yaml
```

## ⚙️ Configuration Details

### StagGAN Configuration (`stag_gan.yaml`)

```yaml
class_path: lit.models.stag_gan.StagGAN
init_args:
  # Architecture
  model_dim: 1024 # SONAR embedding dimension
  state_dim: 2048 # STAG internal state dimension
  num_heads: 32 # Number of attention heads
  num_layers: 1 # Number of STAG layers

  # StyleGAN2 components
  z_dim: 512 # Noise dimension
  w_dim: 512 # Style code dimension
  mapping_layers: 8 # Mapping network depth

  # Training parameters
  g_lr: 0.0001 # Generator learning rate
  d_lr: 0.0004 # Discriminator learning rate (4x generator)
  lambda_gp: 10.0 # Gradient penalty weight
```

### Key Training Parameters

#### GAN Training (`train_stag_gan.yaml`)

- **Epochs**: 100 (can be adjusted based on convergence)
- **Precision**: 16-mixed (for memory efficiency)
- **Gradient Clipping**: 1.0 (for stability)
- **Checkpointing**: Every 2000 steps
- **Early Stopping**: Based on discriminator score difference

#### Sequence Training (`train_stag_cm_pretrained.yaml`)

- **Epochs**: 50 (usually fewer needed after pre-training)
- **Learning Rates**: Muon (0.002) + Adam (0.0001)
- **Checkpointing**: Every 1000 steps
- **Early Stopping**: Based on validation loss

## 📊 Monitoring Training

### TensorBoard Logs

```bash
# Monitor GAN training
tensorboard --logdir lightning_logs/stag_gan_pretraining

# Monitor sequence training
tensorboard --logdir lightning_logs/stag_cm_pretrained_experiment
```

### Key Metrics to Watch

#### GAN Training

- **`d_loss`** - Discriminator loss (should stabilize)
- **`g_loss`** - Generator loss (should decrease initially)
- **`real_score`** vs **`fake_score`** - Score gap indicates discrimination quality
- **`gp`** - Gradient penalty (should be small)

#### Sequence Training

- **`train_loss`** / **`val_loss`** - MSE reconstruction loss
- **Learning rates** - Should follow the scheduler

## 🎛️ Hyperparameter Tuning

### For Better GAN Training

```yaml
# Increase discriminator strength
d_lr: 0.0008 # Higher discriminator LR
n_critic: 2 # More discriminator steps per generator step

# Adjust architecture
num_layers: 2 # Deeper STAG model
mapping_layers: 12 # Deeper mapping network

# Training stability
lambda_gp: 20.0 # Stronger gradient penalty
```

### For Better Sequence Training

```yaml
# Learning rates
muon_lr: 0.001 # Lower for more stable training
scalar_lr: 0.00005 # Lower Adam learning rate

# Architecture
sequence_length: 32 # Longer sequences
batch_size: 16 # Smaller batches for longer sequences
```

## 🔍 Evaluation and Generation

### Testing Generation Quality

```python
from lit.models.stag_gan import StagGAN
from lit.models.stag_cm import StagCM

# Load models
gan_model = StagGAN.load_from_checkpoint("path/to/gan/checkpoint.ckpt")
stagcm_model = StagCM.load_from_checkpoint("path/to/stagcm/checkpoint.ckpt")

# One-shot generation
embeddings = gan_model.sample_embeddings(num_samples=10)

# Sequence generation
sequences = stagcm_model.generate_with_initial_state(batch_size=5, max_length=10)

# Style-controlled generation (if pre-trained)
import torch
z = torch.randn(5, 512)
style_sequences = stagcm_model.generate_from_style(z, max_length=10)
```

### Quality Metrics

- **Embedding norms** - Should be similar to real SONAR embeddings (~1.0)
- **Diversity** - Generated embeddings should be diverse
- **Sequence coherence** - Autoregressive sequences should be smooth

## 🛠️ Troubleshooting

### Common Issues

#### GAN Training

- **Mode collapse**: Increase `lambda_gp`, try different `d_lr/g_lr` ratios
- **Training instability**: Lower learning rates, increase gradient clipping
- **Poor sample quality**: Increase `mapping_layers`, try deeper discriminator

#### Sequence Training

- **High validation loss**: Lower learning rates, increase regularization
- **Slow convergence**: Ensure proper pre-trained checkpoint loading
- **Memory issues**: Reduce `batch_size` or `sequence_length`

### Data Requirements

- **SONAR embeddings**: Ensure data is properly normalized
- **Sufficient data**: Need enough diversity for good GAN training
- **Validation split**: Use separate validation data for proper evaluation

## 📈 Expected Results

### After GAN Pre-training

- Generated embeddings should have realistic norms (~0.8-1.2)
- Discriminator should achieve stable real/fake score difference
- Style interpolation should produce smooth transitions

### After Sequence Training

- Lower autoregressive loss compared to training from scratch
- Better sequence coherence and continuity
- Faster convergence due to pre-trained initial state

## 🎯 Next Steps

1. **Experiment with architectures**: Try different `num_layers`, `state_dim`
2. **Scale up training**: Use larger datasets, longer training
3. **Downstream applications**: Use for data augmentation, semantic search
4. **Advanced techniques**: Try progressive growing, style mixing

## 📚 References

- StyleGAN2 techniques adapted for vector embeddings
- WGAN-GP for stable adversarial training
- Transfer learning from generative pre-training
