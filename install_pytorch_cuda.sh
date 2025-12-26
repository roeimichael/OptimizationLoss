#!/bin/bash

# PyTorch CUDA Installation Script
# This script installs PyTorch with CUDA support

echo "==================================================================="
echo "PyTorch CUDA Installation"
echo "==================================================================="
echo ""

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""

    # Ask for CUDA version
    echo "Select CUDA version:"
    echo "1) CUDA 11.8 (recommended for most systems)"
    echo "2) CUDA 12.1 (for newer GPUs)"
    echo "3) CPU only (no GPU acceleration)"
    read -p "Enter choice [1-3]: " choice

    case $choice in
        1)
            echo "Installing PyTorch with CUDA 11.8..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        2)
            echo "Installing PyTorch with CUDA 12.1..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ;;
        3)
            echo "Installing PyTorch (CPU only)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
        *)
            echo "Invalid choice. Installing CPU version by default..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
else
    echo "⚠ No NVIDIA GPU detected. Installing CPU-only version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Installing other dependencies..."
pip install tqdm numpy matplotlib pandas scikit-learn

echo ""
echo "==================================================================="
echo "Verifying Installation"
echo "==================================================================="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('Running on CPU (no CUDA support)')
"

echo ""
echo "Installation complete!"
