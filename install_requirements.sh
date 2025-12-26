#!/bin/bash

echo "======================================================================="
echo "Installing PyTorch with CUDA Support + Other Dependencies"
echo "======================================================================="
echo ""

# Step 1: Check for GPU
echo "Step 1: Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    CUDA_AVAILABLE=true
else
    echo "⚠ No NVIDIA GPU detected (nvidia-smi not found)"
    echo "  Installing CPU-only version..."
    CUDA_AVAILABLE=false
fi
echo ""

# Step 2: Install PyTorch
echo "Step 2: Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU-only version)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo ""

# Step 3: Install other dependencies
echo "Step 3: Installing other dependencies..."
pip install tqdm numpy matplotlib pandas scikit-learn
echo ""

# Step 4: Verify installation
echo "======================================================================="
echo "Verification"
echo "======================================================================="
python verify_gpu.py
