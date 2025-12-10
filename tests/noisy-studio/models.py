from dataclasses import dataclass

@dataclass
class NoiseParams:
    """Data structure holding all noise generation parameters."""
    impl: str
    seed: int
    frequency: float
    amplitude: float
    octaves: int
    persistence: float
    offset_x: int
    offset_y: int

    def get_offset_str(self) -> str:
        return f"{self.offset_x},{self.offset_y}"