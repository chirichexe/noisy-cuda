import sys
import argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def parse_arguments():
    parser = argparse.ArgumentParser(description="Noisy Cuda")
    
    # Positional argument for binary path  
    parser.add_argument(
        "bin_path", 
        type=str, 
        help="Path to the Noisy Cuda executable"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Check if file exists
    executable_path = Path(args.bin_path).resolve()
    if not executable_path.exists() or not executable_path.is_file():
        print(f"Error: {executable_path} not found")
        sys.exit(1)

    # Create application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow(executable_path=executable_path)
    window.show()
    
    sys.exit(app.exec())