import time
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

controller = GlobalGPUController(interval=10, vram_to_keep=2000)
controller.keep()

time.sleep(10)
controller.release()
print("done")
