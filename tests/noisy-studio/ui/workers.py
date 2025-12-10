from PyQt6.QtCore import QThread, pyqtSignal
from models import NoiseParams
from backend import NoiseGenerator

class GenerationWorker(QThread):
    """Background thread to handle external process execution."""
    
    finished = pyqtSignal(float) # Emits execution time
    error = pyqtSignal(str)      # Emits error message

    def __init__(self, params: NoiseParams):
        super().__init__()
        self.params = params

    def run(self):
        try:
            duration = NoiseGenerator.run(self.params)
            self.finished.emit(duration)
        except Exception as e:
            self.error.emit(str(e))