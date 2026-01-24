import subprocess
import time
from pathlib import Path
from config import OUTPUT_FILE
from models import NoiseParams

class NoiseGenerator:
    def __init__(self, executable_path):
        self.executable_path = executable_path
    
    def run(self, params: NoiseParams) -> float:
        """
        Executes the binary. Returns elapsed time in ms.
        Raises specific exceptions on failure.
        """
        if not self.executable_path.exists():
            raise FileNotFoundError(f"Binary not found at: {self.executable_path}")
        
        print(f"Running noise generation with params: {params}")

        cmd = [
            str(self.executable_path),
            str(params.seed),
            "--output", str(OUTPUT_FILE),
            "--size", f"{params.width}x{params.height}",
            "--format", "ppm",
            "--frequency", str(params.frequency),
            "--amplitude", str(params.amplitude),
            "--persistence", str(params.persistence),
            "--lacunarity", str(params.lacunarity),
            "--octaves", str(params.octaves),
            "--offset", params.get_offset_str()
        ]

        start_t = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_t = time.perf_counter()

        if result.returncode != 0:
            error_output = result.stderr if result.stderr else result.stdout
            error_msg = f"Process failed with code {result.returncode}:\n{error_output}"
            print("Error during noise generation:")
            print(error_msg)
            raise RuntimeError(error_msg)

        return (end_t - start_t) * 1000