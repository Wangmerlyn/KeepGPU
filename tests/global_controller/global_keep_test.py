import time
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

controler = GlobalGPUController(interval=10, vram_to_keep=2000)
controler.keep()

time.sleep(10)
controler.release()
print("done")
