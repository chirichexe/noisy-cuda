from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QMessageBox, QLabel, QDialog, QVBoxLayout, QTextEdit, QPushButton
from PyQt6.QtCore import Qt
from config import DARK_THEME
from models import NoiseParams
from .components import ControlPanel, ImageViewer
from .workers import GenerationWorker

class ErrorDialog(QDialog):
    """Custom error dialog with dark theme."""
    def __init__(self, parent, error_msg: str):
        super().__init__(parent)
        self.setWindowTitle("Error")
        self.setModal(True)
        self.resize(500, 300)
        self.setStyleSheet(DARK_THEME)
        
        layout = QVBoxLayout(self)
        
        # Error text display
        error_text = QTextEdit()
        error_text.setObjectName("ErrorDetails")
        error_text.setText(error_msg)
        error_text.setReadOnly(True)
        layout.addWidget(error_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class MainWindow(QMainWindow):
    def __init__(self, executable_path):
        super().__init__()
        self.setWindowTitle("Noisy Studio")
        self.resize(1200, 800)
        self.setStyleSheet(DARK_THEME)
        
        # Store executable path
        self.executable_path = executable_path

        # State variables
        self.is_loading = False

        # UI Setup
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.sidebar = ControlPanel()
        self.viewer = ImageViewer()
        
        # Status Bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        layout.addWidget(self.sidebar, 1)
        layout.addWidget(self.viewer, 4)

        # Signals ------------------------------------------------- 
        self.sidebar.btn_gen.clicked.connect(self.trigger_generation)
        
        # Connect offset changes for real-time preview
        self.sidebar.spin_offset_x.valueChanged.connect(self.on_offset_changed)
        self.sidebar.spin_offset_y.valueChanged.connect(self.on_offset_changed)
        
        # Update move_step constraints
        self.sidebar.spin_width.valueChanged.connect(self.update_move_step_limits)
        self.sidebar.spin_height.valueChanged.connect(self.update_move_step_limits)

        # Initial render
        self.trigger_generation()

    def update_move_step_limits(self):
        """Update move_step limits based on image dimensions."""
        max_dim = max(self.sidebar.spin_width.value(), self.sidebar.spin_height.value())
        self.sidebar.spin_move_step.setMaximum(max_dim)

    def get_current_params(self) -> NoiseParams:
        """
        Collects parameters including size and manual offset fields.
        """
        return NoiseParams(
            width=self.sidebar.spin_width.value(),
            height=self.sidebar.spin_height.value(),
            seed=int(self.sidebar.spin_seed.value()),
            frequency=self.sidebar.spin_freq.value(),
            amplitude=self.sidebar.spin_amp.value(),
            octaves=self.sidebar.spin_octaves.value(),
            persistence=self.sidebar.spin_persist.value(),
            lacunarity=self.sidebar.spin_lacunarity.value(),
            offset_x=self.sidebar.spin_offset_x.value(),  
            offset_y=self.sidebar.spin_offset_y.value() 
        )

    def trigger_generation(self):
        if self.is_loading:
            return
        self.is_loading = True
        self.status_label.setText("Processing...")
        
        try:
            params = self.get_current_params()
            
            # Create and run worker thread
            self.worker = GenerationWorker(params, self.executable_path)
            self.worker.finished.connect(self.on_generation_success)
            self.worker.error.connect(self.on_generation_error)
            self.worker.start()
        except Exception as e:
            self.on_generation_error(str(e))

    def on_offset_changed(self):
        """Real-time preview regeneration when offset changes with WASD."""
        if self.is_loading:
            return
        self.trigger_generation()

    def on_generation_success(self, duration):
        self.is_loading = False
        self.viewer.load_image()
        ox = self.sidebar.spin_offset_x.value()
        oy = self.sidebar.spin_offset_y.value()
        self.status_label.setText(f"Completed in {duration:.2f}ms | Offset: {ox}, {oy}")

    def on_generation_error(self, err_msg):
        self.is_loading = False
        self.status_label.setText("Error")
        error_dialog = ErrorDialog(self, err_msg)
        error_dialog.exec()

    def keyPressEvent(self, event):
        key = event.key()
        step = self.sidebar.spin_move_step.value()
        current_x = self.sidebar.spin_offset_x.value()
        current_y = self.sidebar.spin_offset_y.value()
        
        if key == Qt.Key.Key_W:
            self.sidebar.spin_offset_y.setValue(current_y - step)
        elif key == Qt.Key.Key_S:
            self.sidebar.spin_offset_y.setValue(current_y + step)
        elif key == Qt.Key.Key_A:
            self.sidebar.spin_offset_x.setValue(current_x - step)
        elif key == Qt.Key.Key_D:
            self.sidebar.spin_offset_x.setValue(current_x + step)
        else:
            super().keyPressEvent(event)