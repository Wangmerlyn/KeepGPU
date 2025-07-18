import time
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController

ctrl = CudaGPUController(rank=1, interval=1, vram_to_keep="20480MiB")

with ctrl:
    print("GPU kept busy for 10 seconds.")
    time.sleep(10)
    print("GPU released.")
print("Test completed successfully.")
