class BaseGPUController:
    def __init__(self, vram_to_keep: int, interval: float):
        """
        Base class for GPU controllers.

        Args:
            vram_to_keep (int): Amount of VRAM (in MB) to keep free.
            interval (int): Time interval (in seconds) for checks or actions.
        """
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
        Method to keep the specified amount of VRAM free.
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
        Asynchronous method to keep the specified amount of VRAM free.
        This is a placeholder for subclasses to implement their logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def _rest(self):
        """
        Asynchronous method to rest or pause the controller.
        This is a placeholder for subclasses to implement their logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
