# Server performance diagnostic script.
# Run on the server to identify training bottlenecks.
# Usage: python diagnose_server.py

import os
import sys
import time
import platform

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_environment():
    section("ENVIRONMENT")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch built with): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB, compute {props.major}.{props.minor})")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    else:
        print("WARNING: CUDA NOT AVAILABLE - training will be on CPU!")
        print(f"  torch.cuda._is_compiled(): {torch.cuda._is_compiled() if hasattr(torch.cuda, '_is_compiled') else 'N/A'}")
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("  nvidia-smi works but PyTorch can't see GPUs - likely CPU-only PyTorch installed!")
                print(result.stdout[:500])
            else:
                print("  nvidia-smi also failed - no GPU drivers?")
        except Exception as e:
            print(f"  nvidia-smi check failed: {e}")


def check_gpu_compute():
    import torch
    if not torch.cuda.is_available():
        print("\n  SKIPPED (no CUDA)")
        return

    section("GPU COMPUTE BENCHMARK")
    device = torch.device('cuda')

    torch.cuda.synchronize()
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    tflops = (2 * 4096**3 * 100) / elapsed / 1e12
    print(f"MatMul 4096x4096 x100: {elapsed:.2f}s ({tflops:.2f} TFLOPS)")
    print(f"  (Expected: ~0.5-2s on modern GPUs, >10s = something wrong)")

    print(f"\nAMP support:")
    print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
    try:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        print(f"  FP16 autocast: OK")
    except Exception as e:
        print(f"  FP16 autocast: FAILED - {e}")


def check_data_loading():
    section("DATA LOADING")

    data_dirs = [
        'data/dermmnist/slice_1',
        'data/dermmnist/slice_2',
        'data/dermmnist',
    ]
    found = None
    for d in data_dirs:
        if os.path.exists(d):
            found = d
            break

    if not found:
        print("No data directory found - skipping data load test")
        return

    print(f"Data dir: {found}")

    import subprocess
    try:
        result = subprocess.run(['df', '-Th', found], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"Filesystem:\n{result.stdout}")
    except Exception:
        pass

    import numpy as np
    npy_files = [f for f in os.listdir(found) if f.endswith('.npy')]
    print(f"Files: {npy_files}")

    if npy_files:
        test_file = os.path.join(found, npy_files[0])
        file_size = os.path.getsize(test_file) / (1024**2)
        print(f"\nLoading {npy_files[0]} ({file_size:.1f} MB)...")

        start = time.time()
        arr = np.load(test_file)
        elapsed = time.time() - start
        print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"  Load time: {elapsed:.3f}s ({file_size/elapsed:.1f} MB/s)")
        print(f"  (Expected: <1s for local SSD, >5s = likely NFS or HDD)")

        start = time.time()
        arr2 = np.load(test_file)
        elapsed2 = time.time() - start
        print(f"  Reload (cached): {elapsed2:.3f}s")


def check_dataloader_throughput():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    section("DATALOADER THROUGHPUT")

    X = torch.randn(8000, 3, 224, 224)
    y = torch.randint(0, 7, (8000,))
    dataset = TensorDataset(X, y)

    for nw in [0, 2, 4]:
        loader = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=nw, pin_memory=torch.cuda.is_available())
        start = time.time()
        for batch_X, batch_y in loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()
        elapsed = time.time() - start
        batches = len(loader)
        print(f"  num_workers={nw}: {elapsed:.2f}s for {batches} batches ({batches/elapsed:.1f} batches/s)")


def check_mini_training():
    import torch
    import torch.nn as nn

    section("MINI TRAINING BENCHMARK")

    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)")
        return

    device = torch.device('cuda')

    from torchvision import models
    model = models.mobilenet_v3_large(weights=None, num_classes=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_fake = torch.randn(256, 3, 224, 224)
    y_fake = torch.randint(0, 7, (256,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_fake, y_fake),
        batch_size=64, shuffle=True)

    torch.backends.cudnn.benchmark = True

    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None

    model.train()
    torch.cuda.synchronize()
    start = time.time()

    for epoch in range(5):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = criterion(model(bx), by)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"MobileNetV3 x 5 epochs x 256 samples (batch=64): {elapsed:.2f}s")
    print(f"  Per epoch: {elapsed/5:.2f}s")
    print(f"  (Expected: <3s total on modern GPU, >30s = problem)")

    mem_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"  Peak GPU memory: {mem_allocated:.0f} MB")


def check_nvidia_smi():
    section("GPU UTILIZATION (nvidia-smi)")
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  {'GPU':>3} | {'Name':>20} | {'GPU%':>5} | {'Mem%':>5} | {'MemUsed':>8} | {'MemTotal':>8} | {'Temp':>5} | {'Power':>6}")
            print(f"  {'-'*3}-+-{'-'*20}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+-{'-'*6}")
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    print(f"  {parts[0]:>3} | {parts[1]:>20} | {parts[2]:>4}% | {parts[3]:>4}% | {parts[4]:>6}MB | {parts[5]:>6}MB | {parts[6]:>4}C | {parts[7]:>5}W")
        else:
            print(f"  nvidia-smi failed: {result.stderr}")
    except FileNotFoundError:
        print("  nvidia-smi not found")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == '__main__':
    print("OptimizationLoss Server Diagnostics")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    check_environment()
    check_nvidia_smi()
    check_gpu_compute()
    check_data_loading()
    check_dataloader_throughput()
    check_mini_training()

    section("SUMMARY")
    print("If GPU compute is fast but training is slow:")
    print("  -> Data loading is the bottleneck (NFS, slow disk, bad num_workers)")
    print("If GPU compute is slow:")
    print("  -> Check CUDA version, GPU driver, or thermal throttling")
    print("If CUDA not available:")
    print("  -> Install GPU PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print()
