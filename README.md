# Super-HQViT: High-Quality Vision Transformer for Tiny ImageNet

Super-HQViT is an optimized Vision Transformer (ViT) model tailored for the Tiny ImageNet dataset. It leverages Neural Architecture Search (NAS), mixed precision training, and distributed training to deliver high performance and scalability.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Using Docker](#using-docker)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  
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
   git clone https://github.com/yourusername/Super-HQViT.git
   cd Super-HQViT
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
