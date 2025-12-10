from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QMessageBox, QLabel
from PyQt6.QtCore import Qt
from config import DARK_THEME, OFFSET_INCREMENT
from models import NoiseParams
from .components import ControlPanel, ImageViewer
from .workers import GenerationWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Noise Studio")
        self.resize(1200, 800)
        self.setStyleSheet(DARK_THEME)

        # State
        self.offset_x = 0
        self.offset_y = 0
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

        # Signals
        self.sidebar.btn_gen.clicked.connect(self.trigger_generation)
        
        # Initial render
        self.trigger_generation()

    def get_current_params(self) -> NoiseParams:
        return NoiseParams(
            impl=self.sidebar.combo_impl.currentText(),
            seed=self.sidebar.spin_seed.value(),
            frequency=self.sidebar.spin_freq.value(),
            amplitude=self.sidebar.spin_amp.value(),
            octaves=self.sidebar.spin_octaves.value(),
            persistence=self.sidebar.spin_persist.value(),
            offset_x=self.offset_x,
            offset_y=self.offset_y
        )

    def trigger_generation(self):
        if self.is_loading: return
        self.is_loading = True
        self.status_label.setText("Processing...")
        
        params = self.get_current_params()
        
        # Create and run worker thread
        self.worker = GenerationWorker(params)
        self.worker.finished.connect(self.on_generation_success)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_success(self, duration):
        self.is_loading = False
        self.viewer.load_image()
        self.status_label.setText(f"Done in {duration:.2f}ms | Offset: {self.offset_x}, {self.offset_y}")

    def on_generation_error(self, err_msg):
        self.is_loading = False
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Generation Error", err_msg)

    def keyPressEvent(self, event):
        key = event.key()
        needs_update = False
        
        if key == Qt.Key.Key_W:
            self.offset_y -= OFFSET_INCREMENT
            needs_update = True
        elif key == Qt.Key.Key_S:
            self.offset_y += OFFSET_INCREMENT
            needs_update = True
        elif key == Qt.Key.Key_A:
            self.offset_x -= OFFSET_INCREMENT
            needs_update = True
        elif key == Qt.Key.Key_D:
            self.offset_x += OFFSET_INCREMENT
            needs_update = True
            
        if needs_update:
            self.trigger_generation()
        else:
            super().keyPressEvent(event)