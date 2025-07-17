import time
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController

controller = CudaGPUController(
    rank=0, interval=1.0, matmul_iterations=1000, vram_to_keep=2**32, busy_threshold=10
)

controller.keep()
# do some CPU-only work here
time.sleep(10)  # simulate CPU work
controller.release()
# now the GPU is free for real work
