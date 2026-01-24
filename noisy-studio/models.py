from dataclasses import dataclass

@dataclass
class NoiseParams:
    """Data structure holding all noise generation parameters."""
    
    # output settings
    width: int
    height: int
    
    # perlin settings
    seed: int
    frequency: float
    amplitude: float
    octaves: int
    persistence: float
    lacunarity: float
    
    # offset settings
    offset_x: int
    offset_y: int

    def get_offset_str(self) -> str:
        return "{},{}".format(int(self.offset_x), int(self.offset_y))