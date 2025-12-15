#!/bin/bash
# Installation script for GPU-accelerated CSP+LDA experiment
# For Lambda Labs A100 instance with CUDA 12.x

echo "=================================================="
echo "Installing GPU Dependencies for CSP+LDA Experiment"
echo "=================================================="
echo ""

# Check CUDA version
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "✓ CUDA version: $CUDA_VERSION"
else
    echo "⚠ Warning: nvcc not found in PATH"
fi

# Check NVIDIA driver
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠ Warning: nvidia-smi not found"
fi

echo "Installing Python packages..."
echo ""

# Install CuPy for GPU-accelerated NumPy operations
# Using CUDA 12.x version
echo "Installing CuPy (GPU-accelerated NumPy)..."
pip install cupy-cuda12x --no-cache-dir

# Install tqdm for progress bars
echo ""
echo "Installing tqdm..."
pip install tqdm

# Verify installation
echo ""
echo "Verifying CuPy installation..."
python3 -c "
import cupy as cp
print('✓ CuPy installed successfully')
print(f'  CuPy version: {cp.__version__}')
print(f'  CUDA available: {cp.cuda.is_available()}')
if cp.cuda.is_available():
    print(f'  Device: {cp.cuda.Device()}')
    print(f'  Compute capability: {cp.cuda.Device().compute_capability}')
    mempool = cp.get_default_memory_pool()
    print(f'  Total GPU memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB')
"

echo ""
echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "To run the GPU-accelerated experiment:"
echo "  python csp_lda_experiment_gpu.py --data_path /path/to/data.npz"
echo ""
echo "To force CPU-only mode:"
echo "  python csp_lda_experiment_gpu.py --data_path /path/to/data.npz --no-gpu"
echo ""