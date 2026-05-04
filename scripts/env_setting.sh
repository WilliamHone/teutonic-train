#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "=== Starting Teutonic Train Setup ==="

# Create main project directory
mkdir -p ~/teutonic
cd ~/teutonic

# Configure Git globally
git config --global user.name "WilliamHone"
git config --global user.email "williamhone136807@outlook.com"

# Clone the repository
git clone https://github.com/WilliamHone/teutonic-train.git

# Create directory structure within the repository
cd teutonic-train
mkdir -p datasets datasets_eval checkpoints/VIII merged/VIII

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install uv package manager
pip install uv

# Install PyTorch with CUDA 12.8 support
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install additional training libraries
uv pip install peft trl liger-kernel

# Install remaining packages via pip
pip install kernels wandb

# Install PM2 process manager globally via npm
npm i -g pm2

# Create shared memory directory structure for fast data access
mkdir -p /dev/shm/teutonic/models
mkdir -p /dev/shm/teutonic/datasets
mkdir -p /dev/shm/teutonic/datasets_2nd
mkdir -p /dev/shm/teutonic/datasets_eval

# Hugging Face authentication
echo ""
echo "=== Hugging Face Authentication ==="
hf auth logout 2>/dev/null || true
echo "Please enter your Hugging Face token (input will be hidden):"
read -s HF_TOKEN
echo ""  # newline after hidden input
echo "$HF_TOKEN" | hf auth login --token-stdin
unset HF_TOKEN  # Clear token from memory

# Weights & Biases authentication
echo ""
echo "=== Weights & Biases Authentication ==="
echo "Please enter your WandB API key (input will be hidden):"
read -s WANDB_TOKEN
echo ""  # newline after hidden input
wandb login --relogin "$WANDB_TOKEN"
unset WANDB_TOKEN  # Clear token from memory

echo ""
echo "=== Setup complete! ==="
echo "✓ Virtual environment is active: .venv"
echo "✓ Directory structure created"
echo "✓ Dependencies installed"
echo "✓ Authentication configured"
echo ""
echo "Next steps:"
echo "  cd ~/teutonic/teutonic-train"
echo "  source .venv/bin/activate  # if starting a new shell"
echo "  # Proceed with your training workflow"