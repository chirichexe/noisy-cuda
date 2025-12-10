from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
EXECUTABLE_PATH = Path("../../build/cuda/noisy_cuda").resolve()
OUTPUT_FILE = BASE_DIR / "temp_noise_frame.ppm"

# Defaults
DEFAULT_SIZE = "1024x1024"
OFFSET_INCREMENT = 128

# Styles
DARK_THEME = """
QMainWindow { background-color: #2b2b2b; }
QWidget { 
    color: #e0e0e0; 
    font-family: 'Segoe UI', 
    sans-serif; 
    font-size: 14px; }
QGroupBox { 
    border: 1px solid #555; 
    border-radius: 5px;
    margin-top: 20px;
    font-weight: bold; }
QGroupBox::title { 
    subcontrol-origin: margin; 
    left: 10px; 
    padding: 0 3px; 
    }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { 
    background-color: #3e3e3e; border: 1px solid #555; border-radius: 3px; padding: 4px; color: #fff; 
}
QPushButton { background-color: #4a90e2; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #357abd; }
QLabel#ImageContainer { border: 2px solid #1a1a1a; background-color: #000; }
"""