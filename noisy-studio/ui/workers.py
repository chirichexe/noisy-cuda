from PyQt6.QtCore import QThread, pyqtSignal
from models import NoiseParams
from backend import NoiseGenerator

class GenerationWorker(QThread):
    """Background thread to handle external process execution."""
    
    finished = pyqtSignal(float) # Emits execution time
    error = pyqtSignal(str)      # Emits error message

    def __init__(self, params: NoiseParams, executable_path):
        super().__init__()
        self.params = params
        self.executable_path = executable_path

    def run(self):
        try:
            duration = NoiseGenerator(self.executable_path).run(self.params)
            self.finished.emit(duration)
        except Exception as e:
            # Emit the complete error message with type info
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.error.emit(error_msg)