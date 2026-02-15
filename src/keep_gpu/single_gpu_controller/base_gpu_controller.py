from typing import Union

from keep_gpu.utilities.humanized_input import parse_size


class BaseGPUController:
    def __init__(self, vram_to_keep: Union[int, str], interval: float):
        """
        Base class for GPU controllers.

        Args:
            vram_to_keep (int or str): Amount of VRAM to keep busy. Accepts integers
                (tensor element count) or human strings like "1GiB" (converted to
                element count for float32 tensors).
            interval (float): Time interval (in seconds) between keep-alive cycles.
        """
        if isinstance(vram_to_keep, str):
            vram_to_keep = parse_size(vram_to_keep)
        elif not isinstance(vram_to_keep, int):
            raise TypeError(
                f"vram_to_keep must be str or int, got {type(vram_to_keep)}"
            )
        self.vram_to_keep = vram_to_keep
        self.interval = interval

    def monitor(self):
        """
        Method to monitor GPU state.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def keep(self):
        """
        Method to keep the specified amount of VRAM busy/occupied.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def rest(self):
        """
        Method to rest or pause the controller.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def _keep(self):
        """
        Asynchronous method to keep the specified amount of VRAM busy/occupied.
        This is a placeholder for subclasses to implement their logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def _rest(self):
        """
        Asynchronous method to rest or pause the controller.
        This is a placeholder for subclasses to implement their logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
