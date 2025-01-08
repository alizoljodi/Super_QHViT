# Super_QHViT

Deploying Hybrid Vision Transformers (H-ViTs) across diverse hardware platforms poses significant challenges due to varying constraints and resource requirements inherent to each device. To overcome these obstacles, we introduce Super-QHViT, an elastic quantized H-ViT supernet that generates multiple architectures with different mixed-precision policies. This design enables efficient adaptation to different hardware environments without the need for extensive retraining.

Super-QHViT is trained using a novel method called Super Sandwich training, which employs a two-level knowledge distillation approach. This technique combines logit-based and feature-based distillation to enhance the performance of each subnet within the supernet. Our supernet is highly configurable, featuring customizable CNN and ViT blocks and supporting various quantization bit-widths.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Using Docker](#using-docker)
- [Usage](#usage)
  - [Training](#training)
  
## Features

- **Vision Transformer (ViT):** Advanced architecture for image classification.
- **Neural Architecture Search (NAS):** Automated architecture optimization.
- **Distributed Training:** Multi-GPU support using PyTorch DDP.
- **Mixed Precision Training:** Faster training with reduced memory usage.
- **Gradient Accumulation:** Efficient training with large batch sizes.
- **Model Checkpointing:** Save and resume training seamlessly.
- **Exponential Moving Average (EMA):** Stabilize training and improve generalization.
- **Comprehensive Logging:** Monitor training with TensorBoard and log files.

## Requirements

- **Docker:** Version 20.10 or higher
- **NVIDIA Docker Runtime:** For GPU support

## Installation

### Using Docker

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Super_QHViT.git
   cd Super_QHViT
2. **Build the Docker Image:**
   Ensure that you have a Dockerfile and requirements.txt in the repository.
   ```bash
   docker build -t superhqvit:latest .
3. **Run the Docker Container:**
   
   Ensure you have NVIDIA Docker runtime installed. Run the container with GPU support:
   ```bash
   docker run --gpus all -it \
    -v /path/to/your/data:/workspace/data \
    -v /path/to/your/configs:/workspace/configs \
    -v /path/to/your/output:/workspace/output \
    superhqvit:latest \
    --cfg configs/config.yaml \
    --working-dir ./output
## Usage
### Training
To start training, run the Docker container with the appropriate arguments:
```bash
docker run --gpus all -it \
-v /path/to/your/data:/workspace/data \
-v /path/to/your/configs:/workspace/configs \
-v /path/to/your/output:/workspace/output \
superhqvit:latest \
--cfg configs/config.yaml \
--working-dir ./output
