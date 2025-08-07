# STAG-GAN: StyleGAN2-inspired Pre-training for Sentence Embeddings

This implementation provides StyleGAN2-inspired components for pre-training STAG models on SONAR sentence embeddings, enabling both one-shot generation and sequence modeling.

## Components

### 1. **StagGenerator** (`models/stag/stag_gan.py`)

- **z → w mapping**: MLP maps latent z∈ℝ^Z to disentangled latent w∈ℝ^W
- **Reusable STAG backbone**: Uses existing STAG architecture for sequence processing
- **Learned initial state**: Trainable beginning-of-sequence representation integrated into STAG
- **One-shot generation**: Style vector as single input to STAG with learned initial state

### 2. **StagDiscriminator** (`models/stag/stag_gan.py`)

- **Spectral normalization**: Stabilizes discriminator training
- **Minibatch standard deviation**: Encourages diversity in generated samples
- **MLP architecture**: Simple but effective for embedding discrimination

### 3. **StagGAN** (`lit/models/stag_gan.py`)

- **WGAN-GP training**: Wasserstein GAN with gradient penalty
- **Lightning integration**: Easy training with PyTorch Lightning
- **One-shot generation**: Focused on learning embedding distribution

### 4. **Enhanced StagCM** (`lit/models/stag_cm.py`)

- **Pre-training integration**: Can load pre-trained GAN components
- **Transfer learning**: Seamless transition from GAN pre-training to sequence training
- **Multiple generation modes**: Initial state-based and style-based generation
- **Unified interface**: One class for both fresh training and pre-trained models

## Usage Workflow

### Phase 1: GAN Pre-training

```python
from lit.models.stag_gan import StagGAN

# Initialize GAN model
gan_model = StagGAN(
    model_dim=1024,     # SONAR embedding dimension
    state_dim=2048,
    num_heads=32,
    num_layers=1,
)

# Train on individual sentence embeddings
trainer.fit(gan_model, dataloader)
```

### Phase 2: Transfer to Sequence Training

```python
from lit.models.stag_cm import StagCM

# Load pre-trained components
stagcm_model = StagCM(
    model_dim=1024,
    state_dim=2048,
    num_heads=32,
    num_layers=1,
    use_pretrained_gan=True,
    gan_checkpoint_path="path/to/gan/checkpoint.ckpt",
    freeze_stag_backbone=False,  # Allow fine-tuning
)

# Continue training on sequences
trainer.fit(stagcm_model, sequence_dataloader)
```

### Phase 3: Generation

```python
# One-shot generation (no input required)
embeddings = gan_model.sample_embeddings(num_samples=8)

# Initial state-based sequence generation
sequences = stagcm_model.generate_with_initial_state(batch_size=4, max_length=10)

# Style-controlled generation
z = torch.randn(4, 512)  # Random noise
sequences = stagcm_model.generate_from_style(z, max_length=10)
```

## Key Features

### StyleGAN2 Adaptations for Vectors

- **Mapping Network**: z → w transformation for disentangled latent space
- **Spectral Normalization**: Stabilizes discriminator training
- **Minibatch Stddev**: Encourages sample diversity
- **Integrated Initial State**: Learned BOS state is part of STAG architecture

### STAG Integration

- **Always Learnable Initial State**: Every STAG model has learned initial state
- **Reusable Backbone**: Pre-trained STAG model transfers seamlessly to sequence tasks
- **Unified Interface**: Single StagCM class handles both fresh and pre-trained models
- **Flexible Architecture**: Compatible with existing STAG configurations

## Architecture Details

### Generator Architecture

```
Noise z (512)
    ↓
Mapping Network (8 layers)
    ↓
Style Code w (512)
    ↓
Style Projection → input (1024)
    ↓
STAG(input, learned_initial_state) → embedding (1024)
```

### Training Phases

1. **One-shot Pre-training**: Learn to generate realistic individual embeddings
2. **Sequence Fine-tuning**: Adapt to autoregressive sequence modeling
3. **Unified Training**: Single StagCM class handles both phases

## File Structure

```
models/stag/
├── stag.py                  # Enhanced STAG with learned initial state
├── stag_gan.py              # Generator & Discriminator
lit/models/
├── stag_gan.py              # GAN Lightning module
├── stag_cm.py               # Enhanced StagCM with pre-training support
examples/
├── stag_gan_pretraining_example.py  # Complete workflow example
```

## Dependencies

- PyTorch Lightning
- SONAR (for real sentence embeddings)
- Spectral normalization (built into PyTorch)
- Existing STAG dependencies

## Benefits

### For Research

- **Controllable Generation**: Style codes enable controllable sentence embedding generation
- **Few-shot Learning**: Pre-trained representations improve sample efficiency
- **Disentanglement**: Mapping network learns disentangled embedding space
- **Unified Architecture**: Single interface for all training modes

### For Applications

- **Data Augmentation**: Generate diverse sentence embeddings for training
- **Semantic Interpolation**: Smooth interpolation in embedding space
- **Zero-shot Generation**: Generate embeddings without input text
- **Transfer Learning**: Pre-trained backbone accelerates downstream training
