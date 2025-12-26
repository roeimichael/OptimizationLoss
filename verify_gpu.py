#!/usr/bin/env python3
"""
GPU and CUDA Verification Script
Checks if PyTorch is properly installed with CUDA support
"""

import sys

print("=" * 70)
print("PyTorch and CUDA Verification")
print("=" * 70)
print()

# Check Python version
print(f"Python version: {sys.version}")
print()

# Check PyTorch installation
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except ImportError:
    print("✗ PyTorch is not installed!")
    print("  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

print()
print("-" * 70)
print("CUDA Availability Check")
print("-" * 70)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"✓ CUDA is available!")
    print(f"  CUDA version (PyTorch): {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print()

    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

    print()
    print("-" * 70)
    print("Testing GPU Operations")
    print("-" * 70)

    # Test GPU operations
    try:
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("✓ GPU tensor operations successful!")
        print(f"  Test tensor shape: {z.shape}")
        print(f"  Test tensor device: {z.device}")
    except Exception as e:
        print(f"✗ GPU tensor operations failed: {e}")

else:
    print("✗ CUDA is NOT available!")
    print()
    print("Possible reasons:")
    print("  1. PyTorch was installed without CUDA support (CPU-only version)")
    print("  2. No NVIDIA GPU available on this system")
    print("  3. CUDA drivers are not properly installed")
    print()
    print("To fix:")
    print("  1. Check if you have an NVIDIA GPU:")
    print("     Run: nvidia-smi")
    print()
    print("  2. Reinstall PyTorch with CUDA support:")
    print("     For CUDA 11.8:")
    print("     pip uninstall torch torchvision torchaudio")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("  3. For CUDA 12.1:")
    print("     pip uninstall torch torchvision torchaudio")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print()
print("-" * 70)
print("PyTorch Build Information")
print("-" * 70)
print(f"Built with CUDA: {torch.version.cuda is not None}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.enabled else 'N/A'}")

print()
print("=" * 70)
if cuda_available:
    print("✓ SUCCESS: PyTorch is properly configured with CUDA support!")
else:
    print("✗ WARNING: PyTorch is running in CPU-only mode!")
print("=" * 70)
