from typing import Literal

from msgspec import Struct


type DeviceType = Literal["cpu", "cuda:0", "cuda:1"]


class EnvArgs(Struct):
    """Arguments for host environment"""

    # Seed to reproduce output
    seed: int = 42

    # Accelerator used to run the code
    device: DeviceType = "cuda:0"


class IOArgs(Struct):
    """Arguments for convertor I/O"""

    # Path to the RGB input image
    input: str

    # Path to the character sheet
    chars: str

    # Path to the font file
    font: str

    # Path to the output dir
    output: str

    # Path to create glyphs
    glyphs: str


class ASCIIArgs(Struct):
    """Arguments for adjusting ASCII output"""

    # Size of the font
    size: int = 8

    # Pad ASCII glyphs
    pad: float = 0.25


class SGDAlgorithmArgs(Struct):
    """Arguments for SGD algorithm"""

    # Maximum optimization iterations
    steps: int = 64

    # Error threshold to end optimization
    eps: float = 0.01

    # Convergence rate
    eta: float = 0.75

    # Gumbel-Softmax smoothness factor
    tau: float = 0.75

    # Exponential moving average for solution params
    ema: float = 0.25


class Args(Struct):
    """Convert an RGB Image to ASCII art"""

    # Env arguments
    env: EnvArgs

    # I/O arguments
    io: IOArgs

    # ASCII arguments
    ascii: ASCIIArgs

    # Algorithm arguments
    alg: SGDAlgorithmArgs
