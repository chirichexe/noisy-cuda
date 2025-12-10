from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, 
                             QComboBox, QSpinBox, QDoubleSpinBox, QLabel, QPushButton)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2
from config import OUTPUT_FILE

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # General Settings
        grp_gen = QGroupBox("General")
        frm_gen = QFormLayout()
        self.combo_impl = QComboBox()
        self.combo_impl.addItems(["cuda", "cpp", "simd"])
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(1234)
        frm_gen.addRow("Backend:", self.combo_impl)
        frm_gen.addRow("Seed:", self.spin_seed)
        grp_gen.setLayout(frm_gen)
        layout.addWidget(grp_gen)

        # Noise Settings
        grp_noise = QGroupBox("Parameters")
        frm_noise = QFormLayout()
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setValue(4.0)
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setValue(1.0)
        self.spin_octaves = QSpinBox()
        self.spin_octaves.setValue(4)
        self.spin_persist = QDoubleSpinBox()
        self.spin_persist.setValue(0.5)
        
        frm_noise.addRow("Freq:", self.spin_freq)
        frm_noise.addRow("Amp:", self.spin_amp)
        frm_noise.addRow("Octs:", self.spin_octaves)
        frm_noise.addRow("Pers:", self.spin_persist)
        grp_noise.setLayout(frm_noise)
        layout.addWidget(grp_noise)

        # Button
        self.btn_gen = QPushButton("GENERATE")
        self.btn_gen.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.btn_gen)
        layout.addStretch()

class ImageViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageContainer")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.current_pixmap = None

    def load_image(self):
        if not OUTPUT_FILE.exists(): return
        
        # OpenCV loading for robust PPM support
        cv_img = cv2.imread(str(OUTPUT_FILE), cv2.IMREAD_COLOR)
        if cv_img is None: return

        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(q_img)
        self.update_display()

    def update_display(self):
        if self.current_pixmap:
            scaled = self.current_pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)