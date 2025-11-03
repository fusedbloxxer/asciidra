import imageio.v3 as iio
import matplotlib.pyplot as plt
import msgspec
import numpy.typing as npt
import torchvision.transforms.v2.functional as TF
import tyro

from torch import Tensor
from torchvision.tv_tensors import Image
from tyro.conf import OmitArgPrefixes


class Args(msgspec.Struct):
    """Convert an RGB Image to ASCII art"""

    # Path to the RGB input image
    input: str

    # Path to the character sheet
    chars: str


def main(args: OmitArgPrefixes[Args]) -> None:
    image_raw: npt.NDArray = iio.imread(args.input)
    image_rgb: Image = TF.to_image(image_raw)

    image_gray: Tensor = TF.to_dtype(image_rgb, scale=True)
    image_gray: Tensor = TF.to_grayscale(image_gray).squeeze()

    plt.imshow(image_gray, cmap="grey")
    plt.show()


tyro.cli(main)
