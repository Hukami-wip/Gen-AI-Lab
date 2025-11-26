# Gen-AI-Lab

A personal research laboratory for studying, implementing, and experimenting with generative AI and sequence modeling techniques. 
Built with PyTorch and PyTorch Lightning.

## Overview

This repository serves as a sandbox for:

- **Paper Implementations**: From-scratch implementations of published architectures (Mamba SSM, Transformers) to deeply understand their mechanics
- **Prototyping Notebooks**: Jupyter notebooks for studying concepts, testing ideas, and developing new approaches
- **Training Experiments**: End-to-end pipelines for benchmarking and comparing models on various tasks

## Content

### Model Implementations

| Name | Type | Description | Location |
|------|------|-------------|----------|
| Mamba | Paper impl. | From-scratch SSM with selective scan | `models/mamba/` |
| Transformer | Paper impl. | Standard encoder architecture | `lit/models/transformer_model.py` |
| STAG | Custom | State-gated RNN with attention-like updates | `models/stag/` |
| STAG-GAN | Custom | StyleGAN2-inspired generator for embeddings | `models/stag/stag_gan.py` |

### Notebooks

| Notebook | Topic | Description |
|----------|-------|-------------|
| `attention_is_all_you_need.ipynb` | Transformers | Study of the original attention mechanism |
| `Mamba_MNIST.ipynb` | Mamba | SSM architecture on sequential MNIST |
| `state_gated_recurrent_cell.ipynb` | RNN design | Exploration of gating mechanisms for RNNs |
| `generative_temporal_sequence.ipynb` | Sequence generation | Temporal sequence modeling experiments |
| `sonar_experiment.ipynb` | Embeddings | Working with SONAR sentence embeddings |
| `StagGAN_inference.ipynb` | GAN | Generating sentence embeddings with GAN |
| `StagCM_inference.ipynb` | Context model | Sequence modeling with STAG |

## Installation

```bash
# Clone and setup
git clone <repo-url>
cd Gen-AI-Lab

conda create -n genai python=3.11
conda activate genai
pip install -r requirements.txt

# Optional dependencies
pip install mamba-ssm      # For Mamba SSM
pip install sonar-space    # For SONAR embeddings
```

## Project Structure

```
Gen-AI-Lab/
├── models/                 # Core model implementations
│   ├── mamba/             # Mamba SSM from scratch
│   └── stag/              # Custom STAG architecture
│
├── lit/                    # PyTorch Lightning modules
│   ├── models/            # LightningModule wrappers
│   ├── data/              # DataModules (MQAR, sorting, MNIST, etc.)
│   ├── callbacks/         # Training callbacks
│   └── optimizers/        # Custom optimizers
│
├── configs/                # YAML configurations
│   ├── data/              # Dataset configs
│   ├── model/             # Model configs
│   └── trainer/           # Training configs
│
├── notebooks/              # Study & prototyping notebooks
├── scripts/                # Data preparation utilities
├── docs/                   # Documentation
└── lit_cli.py              # Lightning CLI entry point
```

## Usage

### Training with Lightning CLI

```bash
# Generic pattern
python lit_cli.py fit \
  --config configs/trainer/<trainer>.yaml \
  --config configs/data/<dataset>.yaml \
  --config configs/model/<model>.yaml

# Example: Train on MQAR benchmark
python lit_cli.py fit \
  --config configs/trainer/train_mqar.yaml \
  --config configs/data/mqar.yaml \
  --config configs/model/stag_mqar_small.yaml

# Override parameters
python lit_cli.py fit \
  --config configs/trainer/train_mqar.yaml \
  --config configs/data/mqar.yaml \
  --config configs/model/stag_mqar_small.yaml \
  --trainer.max_epochs 100 \
  --data.batch_size 256
```

### Testing

```bash
python lit_cli.py test \
  --config configs/trainer/train_mqar.yaml \
  --config configs/data/mqar.yaml \
  --config configs/model/stag_mqar_small.yaml \
  --ckpt_path lightning_logs/mqar/version_X/checkpoints/best.ckpt
```

## Available Benchmarks

| Benchmark | Task | DataModule |
|-----------|------|------------|
| MQAR | Multi-query associative recall | `mqar_datamodule.py` |
| Sorting | Sequence sorting | `sorting_datamodule.py` |
| sMNIST | Sequential MNIST classification | `mnist_datamodule.py` |
| SONAR | Sentence embedding modeling | `sonar_parquet_datamodule.py` |

## Monitoring

```bash
tensorboard --logdir lightning_logs/
```

## License

MIT License
