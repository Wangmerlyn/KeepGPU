from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController

ctrl = CudaGPUController(rank=1, interval=10, vram_to_keep=1000*1000)
ctrl.keep()
print("GPU kept busy for 10 seconds.")
import time
time.sleep(10)
ctrl.release()
print("GPU released.")

print("test for 2nd time")
ctrl.keep()
print("GPU kept busy for another 10 seconds.")
time.sleep(10)
ctrl.release()
print("GPU released again.")
print("Test completed successfully.")
# This code snippet is for testing the CudaGPUController functionality.