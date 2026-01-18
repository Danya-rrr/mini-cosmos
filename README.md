# Mini-Cosmos: World Model for Autonomous Driving

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*A compact world model for predicting road scenes and planning actions in autonomous driving simulation*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Results](#results)

</div>

---

## 📋 Overview

Mini-Cosmos is a compact world model designed for autonomous driving that learns to predict future road scenes based on past observations and vehicle actions. The system uses a two-stage approach:

1. **VAE (Variational Autoencoder)** - Compresses 256×256 RGB images into compact 32×32×4 latent representations (64× compression)
2. **World Model (Transformer)** - Predicts future latent states based on 8 past frames and vehicle actions

The model is trained and evaluated on data from the CARLA driving simulator and nuScenes dataset.

## ✨ Features

- **Compact Representation**: 64× image compression with minimal quality loss
- **Temporal Prediction**: Autoregressive prediction of future driving scenes
- **Action Conditioning**: Predictions conditioned on vehicle control inputs (throttle, steering, brake)
- **Real-time Inference**: Live demonstration with CARLA simulator
- **Comprehensive Evaluation**: SSIM, PSNR, MSE, and LPIPS metrics
- **Mixed Precision Training**: FP16 support for faster training on modern GPUs

## 🏗️ Architecture

### VAE Architecture
```
Input: [B, 3, 256, 256]
  ↓
Encoder: Conv + ResBlocks (256→128→64→32)
  ↓
Latent: [B, 4, 32, 32]
  ↓
Decoder: ResBlocks + Upsample (32→64→128→256)
  ↓
Output: [B, 3, 256, 256]
```

### World Model Architecture
```
Context: [B, 8, 4, 32, 32] + Actions: [B, 8, 3]
  ↓
Patch Embedding: 32×32 → 8×8 patches
  ↓
Transformer: 6 layers, 512 hidden dim, 8 heads
  ↓
Prediction: [B, 4, 32, 32]
```

**Key Parameters:**
- VAE: ~12M parameters
- World Model: ~45M parameters
- Total: ~57M parameters

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- CARLA Simulator 0.9.13+ (optional, for data collection and live demo)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mini-cosmos.git
cd mini-cosmos
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install CARLA (optional)**
```bash
# Download CARLA from https://github.com/carla-simulator/carla/releases
# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla
```

### Project Structure
```
mini-cosmos/
├── src/
│   ├── models/
│   │   ├── encoder.py          # VAE encoder
│   │   ├── decoder.py          # VAE decoder
│   │   ├── vae.py              # Complete VAE model
│   │   └── world_model.py      # Temporal transformer
│   └── data/
│       └── dataset.py          # Data loading and augmentation
├── scripts/
│   ├── train_vae.py            # Train VAE
│   ├── train_world_model.py    # Train World Model
│   ├── evaluate_metrics.py     # Compute metrics
│   ├── generate_gif.py         # Generate prediction GIFs
│   ├── generate_summary.py     # Generate summary images
│   ├── visualize_vae.py        # Visualize VAE reconstructions
│   ├── live_demo.py            # Live CARLA demo
│   ├── collect_data.py         # Collect CARLA data
│   └── prepare_nuscenes.py     # Prepare nuScenes data
├── data/                       # Data directory
├── outputs/                    # Outputs (checkpoints, logs, etc.)
└── README.md
```

## 📊 Usage

### 1. Data Collection

**Collect data from CARLA:**
```bash
python scripts/collect_data.py \
    --episodes 50 \
    --frames 500 \
    --output ./data/raw \
    --host localhost \
    --port 2000
```

**Prepare nuScenes data:**
```bash
python scripts/prepare_nuscenes.py
```

### 2. Training

**Train VAE:**
```bash
python scripts/train_vae.py \
    --carla_dir ./data/raw \
    --nuscenes_dir ./data/processed/nuscenes \
    --batch_size 32 \
    --epochs 30 \
    --latent_dim 4 \
    --amp
```

**Train World Model:**
```bash
python scripts/train_world_model.py \
    --vae_checkpoint ./outputs/checkpoints/vae_best.pt \
    --batch_size 8 \
    --epochs 30 \
    --context_length 8 \
    --hidden_dim 512 \
    --amp
```

### 3. Evaluation

**Compute metrics:**
```bash
python scripts/evaluate_metrics.py \
    --vae_checkpoint ./outputs/checkpoints/vae_best.pt \
    --world_model_checkpoint ./outputs/checkpoints/world_model_best.pt \
    --num_samples 100 \
    --num_future 16
```

**Generate prediction GIFs:**
```bash
python scripts/generate_gif.py \
    --num_samples 5 \
    --num_future 16 \
    --with_gt
```

**Generate summary visualization:**
```bash
python scripts/generate_summary.py \
    --num_samples 4 \
    --num_future 60
```

### 4. Live Demo

**Run live prediction with CARLA:**
```bash
python scripts/live_demo.py \
    --carla_host localhost \
    --carla_port 2000 \
    --num_future 4 \
    --show_gt
```

**Controls:**
- `Q` - Quit
- `S` - Screenshot
- `R` - Start/stop recording
- `SPACE` - Pause/resume

## 📈 Results

### VAE Reconstruction Quality

| Metric | Value |
|--------|-------|
| SSIM | 0.85+ |
| PSNR | 25+ dB |
| Compression Ratio | 64× |

### World Model Prediction

| Horizon | SSIM | PSNR | MSE |
|---------|------|------|-----|
| t+1 | 0.82 | 24.5 | 0.015 |
| t+5 | 0.75 | 22.8 | 0.025 |
| t+10 | 0.68 | 21.2 | 0.038 |

*Metrics evaluated on CARLA test set*

## 🛠️ Technical Details

### VAE Training
- **Loss**: MSE reconstruction + KL divergence (β=0.0001)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine annealing
- **Augmentation**: Color jitter, horizontal flip
- **Mixed Precision**: FP16 with gradient scaling

### World Model Training
- **Loss**: MSE on latent space predictions
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: OneCycleLR with 10% warmup
- **Context Length**: 8 frames
- **Patch Size**: 4×4 (64 patches per frame)

### Hardware Requirements
- **Minimum**: 8GB GPU VRAM (RTX 3060)
- **Recommended**: 12GB+ GPU VRAM (RTX 3090, A5000)
- **CPU**: 8+ cores for data loading
- **RAM**: 16GB+
- **Storage**: 50GB+ for datasets

## 📝 Configuration

### VAE Config
```python
VAEConfig(
    in_channels=3,
    latent_dim=4,
    hidden_dims=(64, 128, 256, 256),
    beta=0.0001
)
```

### World Model Config
```python
WorldModelConfig(
    latent_dim=4,
    latent_size=32,
    context_length=8,
    action_dim=3,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    dropout=0.1
)
```

## 🔬 Experiments

The model was trained on:
- **CARLA Data**: 25,000 frames (50 episodes × 500 frames)
- **nuScenes-mini**: 20,000 frames (400 scenes)
- **Total**: ~45,000 training sequences

Training time:
- VAE: ~3 hours (28 epochs, RTX 3090)
- World Model: ~6 hours (24 epochs, RTX 3090)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [CARLA Simulator](https://carla.org/) - Open-source autonomous driving simulator
- [nuScenes Dataset](https://www.nuscenes.org/) - Large-scale autonomous driving dataset
- Inspired by [World Models](https://worldmodels.github.io/) and [DreamerV3](https://danijar.com/project/dreamerv3/)

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:
- **Author**: Daniil
- **GitHub**: [@Danya-rrr](https://github.com/Danya-rrr)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for autonomous driving research

</div>