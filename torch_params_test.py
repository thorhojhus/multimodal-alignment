import time
import torch

# Matrix size and benchmark parameters
N = 4096
FLOPS = N*N*N*2  # For GEMM operations
warmup = 10
iterations = 512
cooldown = 1

# Memory bandwidth test settings
mem_size_gb = 1.0  # Size in GB for memory test
mem_warmup = 5
mem_iterations = 32

def get_gpu_info():
    """Get GPU model name and other details"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{gpu_name} ({gpu_mem:.2f} GB)"
    return "No GPU detected"

def run_compute_benchmark(dtype_name):
    """Run a compute benchmark with high precision mode and specified data type"""
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')  # Use TF32 for float32
    
    dtype = getattr(torch, dtype_name)
    
    # Create random matrices
    b = torch.rand((N, N), dtype=dtype, device="cuda")
    c = torch.rand((N, N), dtype=dtype, device="cuda")
    
    # Warmup
    for _ in range(warmup):
        a = b @ c
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        st = time.perf_counter()
        a = b @ c
        torch.cuda.synchronize()
        times.append(time.perf_counter() - st)
    
    # Calculate performance
    tm = min(times)
    tflops = FLOPS * 1e-12 / tm
    
    print(f"{dtype_name:10s}: {tm*1e6:8.2f} μs, {tflops:7.2f} TFLOPS")
    
    # Cooldown period
    time.sleep(cooldown)
    
    return tflops

def run_amp_benchmark():
    """Run benchmark with Automatic Mixed Precision"""
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    
    # Create FP32 tensors
    b = torch.rand((N, N), dtype=torch.float32, device="cuda")
    c = torch.rand((N, N), dtype=torch.float32, device="cuda")
    
    # Warmup
    for _ in range(warmup):
        with torch.amp.autocast(device_type='cuda'):
            a = b @ c
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        st = time.perf_counter()
        with torch.amp.autocast(device_type='cuda'):
            a = b @ c
        torch.cuda.synchronize()
        times.append(time.perf_counter() - st)
    
    # Calculate performance
    tm = min(times)
    tflops = FLOPS * 1e-12 / tm
    
    print(f"{'amp':10s}: {tm*1e6:8.2f} μs, {tflops:7.2f} TFLOPS")
    
    # Cooldown period
    time.sleep(cooldown)
    
    return tflops

def measure_memory_bandwidth():
    """Measure memory bandwidth in GB/s using tensor operations"""
    torch.cuda.empty_cache()
    
    # Calculate tensor size to match desired memory usage
    num_elements = int(mem_size_gb * 1e9 / 4)  # 4 bytes per float
    
    # For memory bandwidth testing, use flat vectors to ensure
    # contiguous memory access patterns
    x = torch.ones(num_elements, dtype=torch.float32, device="cuda")
    y = torch.ones(num_elements, dtype=torch.float32, device="cuda")
    
    # Bytes moved in each test (read x, y, write z)
    bytes_per_iter = num_elements * 4 * 3  # 3 = 2 reads + 1 write
    
    # Warmup
    for _ in range(mem_warmup):
        z = x + y
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(mem_iterations):
        torch.cuda.synchronize()
        st = time.perf_counter()
        z = x + y
        torch.cuda.synchronize()
        times.append(time.perf_counter() - st)
    
    # Calculate bandwidth
    tm = min(times)
    bandwidth_gbps = bytes_per_iter / tm / 1e9
    
    print(f"\nMemory Bandwidth Test ({mem_size_gb:.1f} GB tensor)")
    print(f"Vector Addition: {bandwidth_gbps:.2f} GB/s")
    
    # Additional memory test: copy operation
    times = []
    for _ in range(mem_iterations):
        torch.cuda.synchronize()
        st = time.perf_counter()
        z = x.clone()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - st)
    
    # Calculate bandwidth (copy is 1 read + 1 write)
    tm = min(times)
    memcpy_bandwidth_gbps = (num_elements * 4 * 2) / tm / 1e9
    
    print(f"Memory Copy:     {memcpy_bandwidth_gbps:.2f} GB/s")

def measure_cpu_gpu_transfer():
    """Measure CPU<->GPU transfer speed in GB/s"""
    torch.cuda.empty_cache()
    
    # Use half the memory size for transfer tests to avoid OOM
    transfer_size_gb = mem_size_gb / 2
    num_elements = int(transfer_size_gb * 1e9 / 4)  # 4 bytes per float
    
    # Create CPU tensor
    x_cpu = torch.ones(num_elements, dtype=torch.float32)
    
    # Warmup
    for _ in range(mem_warmup):
        x_gpu = x_cpu.cuda()
        torch.cuda.synchronize()
        x_back = x_gpu.cpu()
    
    # CPU -> GPU transfer
    times_to_gpu = []
    for _ in range(mem_iterations):
        torch.cuda.synchronize()
        st = time.perf_counter()
        x_gpu = x_cpu.cuda()
        torch.cuda.synchronize()
        times_to_gpu.append(time.perf_counter() - st)
    
    # GPU -> CPU transfer
    times_to_cpu = []
    for _ in range(mem_iterations):
        torch.cuda.synchronize()
        st = time.perf_counter()
        x_back = x_gpu.cpu()
        # No synchronize needed for CPU operations
        times_to_cpu.append(time.perf_counter() - st)
    
    # Calculate bandwidth
    tm_to_gpu = min(times_to_gpu)
    tm_to_cpu = min(times_to_cpu)
    
    bytes_transferred = num_elements * 4
    to_gpu_gbps = bytes_transferred / tm_to_gpu / 1e9
    to_cpu_gbps = bytes_transferred / tm_to_cpu / 1e9
    
    print(f"\nCPU<->GPU Transfer Test ({transfer_size_gb:.1f} GB tensor)")
    print(f"CPU -> GPU:      {to_gpu_gbps:.2f} GB/s")
    print(f"GPU -> CPU:      {to_cpu_gbps:.2f} GB/s")

def main():
    # Print header information first
    print(f"GPU: {get_gpu_info()}")
    print(f"Matrix Size: {N}x{N} ({N*N*4/1e9:.2f} GB per matrix)")
    print("=" * 60)
    
    # Compute benchmarks
    print("Matrix Multiplication Performance:")
    for dtype in ["float32", "float16", "bfloat16"]:
        try:
            run_compute_benchmark(dtype)
        except Exception as e:
            print(f"Error testing {dtype}: {e}")
    
    try:
        run_amp_benchmark()
    except Exception as e:
        print(f"Error testing AMP: {e}")
    
    # Memory bandwidth benchmarks
    try:
        measure_memory_bandwidth()
    except Exception as e:
        print(f"Error in memory bandwidth test: {e}")
    
    # CPU<->GPU transfer speed
    try:
        measure_cpu_gpu_transfer()
    except Exception as e:
        print(f"Error in CPU<->GPU transfer test: {e}")

if __name__ == "__main__":
    main()


# GPU: Tesla V100-SXM2-32GB (34.07 GB)
# Matrix Size: 4096x4096 (0.07 GB per matrix)
# ============================================================
# Matrix Multiplication Performance:
# float32   :  9740.13 μs,   14.11 TFLOPS
# float16   :  1444.73 μs,   95.13 TFLOPS
# bfloat16  : 12978.47 μs,   10.59 TFLOPS
# amp       :  1678.82 μs,   81.87 TFLOPS

# Memory Bandwidth Test (1.0 GB tensor)
# Vector Addition: 830.39 GB/s
# Memory Copy:     817.08 GB/s

# CPU<->GPU Transfer Test (0.5 GB tensor)
# CPU -> GPU:      4.62 GB/s
# GPU -> CPU:      2.82 GB/s

# GPU: NVIDIA A100-PCIE-40GB (42.41 GB)
# Matrix Size: 4096x4096 (0.07 GB per matrix)
# ============================================================
# Matrix Multiplication Performance:
# float32   :  1154.49 μs,  119.05 TFLOPS
# float16   :   563.74 μs,  243.80 TFLOPS
# bfloat16  :   544.49 μs,  252.42 TFLOPS
# amp       :   718.73 μs,  191.22 TFLOPS

# Memory Bandwidth Test (1.0 GB tensor)
# Vector Addition: 1350.12 GB/s
# Memory Copy:     1363.22 GB/s

# CPU<->GPU Transfer Test (0.5 GB tensor)
# CPU -> GPU:      3.99 GB/s
# GPU -> CPU:      2.74 GB/s

# GPU: NVIDIA GeForce RTX 3090 (25.77 GB)
# Matrix Size: 4096x4096 (0.07 GB per matrix)
# ============================================================
# Matrix Multiplication Performance:
# float32   :  4401.10 μs,   31.23 TFLOPS
# float16   :  1966.14 μs,   69.90 TFLOPS
# bfloat16  :  1800.23 μs,   76.35 TFLOPS
# amp       :  2033.68 μs,   67.58 TFLOPS

# Memory Bandwidth Test (1.0 GB tensor)
# Vector Addition: 906.76 GB/s
# Memory Copy:     890.18 GB/s

# CPU<->GPU Transfer Test (0.5 GB tensor)
# CPU -> GPU:      13.74 GB/s
# GPU -> CPU:      7.59 GB/s