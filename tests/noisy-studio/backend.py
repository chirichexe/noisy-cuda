import subprocess
import time
from pathlib import Path
from config import EXECUTABLE_PATH, OUTPUT_FILE, DEFAULT_SIZE
from models import NoiseParams

class NoiseGenerator:
    @staticmethod
    def run(params: NoiseParams) -> float:
        """
        Executes the binary. Returns elapsed time in ms.
        Raises specific exceptions on failure.
        """
        if not EXECUTABLE_PATH.exists():
            raise FileNotFoundError(f"Binary not found at: {EXECUTABLE_PATH}")

        cmd = [
            str(EXECUTABLE_PATH),
            str(params.seed),
            "--output", str(OUTPUT_FILE),
            "--size", DEFAULT_SIZE,
            "--format", "ppm",
            "--frequency", str(params.frequency),
            "--amplitude", str(params.amplitude),
            "--persistence", str(params.persistence),
            "--octaves", str(params.octaves),
            "--offset", params.get_offset_str()
        ]

        start_t = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_t = time.perf_counter()

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr
            )

        return (end_t - start_t) * 1000