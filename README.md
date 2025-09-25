

# Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing (CVPR 2025)

This repository contains the official PyTorch implementation for the paper: "[Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing](https://arxiv.org/abs/2403.16134)" accepted at CVPR 2025.

Our work, BMTNet, introduces a lightweight Mamba-based binary neural network for efficient and high-performance demosaicing of HybridEVS RAW images. We propose a hybrid Binarized Mamba-Transformer architecture to effectively capture both global and local dependencies.

## Installation

We recommend using Miniconda for environment management.

**Step 1. Clone the repository**

```bash
git clone https://github.com/Clausy9/BMTNet.git
cd BMTNet
```

**Step 2. Create and activate the conda environment**


```bash
conda create --name bmtnet python=3.8 -y
conda activate bmtnet
```

**Step 3. Install dependencies**

Install PyTorch according to the official instructions for your specific CUDA version. For example:

```bash
# Example for CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

Then, install the remaining required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:
```
argparse
tqdm
scikit-image
opencv-python
einops
timm
thop
random-fourier-features-pytorch
mamba-ssm==2.2.1
```

***Prerequisites***
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

## Testing

**Step 1. Download pretrained models**

Download the pretrained models from [[Google Drive](https://drive.google.com/file/d/1bH1qR-T7XoTUMzVZlQyfMKFZQ6lb9NmT/view?usp=sharing)] and place them in the `./weights` directory.

**Step 2. Prepare your data**

Place your input images in the `./input` directory.

**Step 3. Run the testing script**

To run the testing script, use the following command. The output will be saved in the `./output` directory.

```bash
python test.py --input_path ./input --weights_path ./weights/bmtnet_model_latest.pth --save_path ./output
```

You can adjust the input and output paths as needed.


