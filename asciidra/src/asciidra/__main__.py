import os
import pathlib
import shutil

from typing import List, Tuple

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL
import PIL.Image
import torch
import torchvision.transforms.v2.functional as TF
import tyro

from matplotlib.axes import Axes
from msgspec.structs import asdict
from PIL import ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from torchvision.tv_tensors import Image
from tyro.conf import OmitArgPrefixes

from .algs import Algorithm, ASCIIConfig, ASCIIOutput, SGDAlgorithm
from .args import Args


def main(args: OmitArgPrefixes[Args]) -> None:
    # Environment
    torch.manual_seed(args.env.seed)
    device = torch.device(args.env.device)

    # Read input image as grayscale and convert to tensor
    image_raw: npt.NDArray = iio.imread(args.io.input, pilmode="RGB").astype(np.uint8)
    image_rgb: Image = TF.to_image(image_raw)
    image_input = TF.to_dtype(image_rgb, scale=True)
    image_input = TF.to_grayscale(image_input)

    # Use charset from disk
    with open(args.io.chars, "r", encoding="utf-8") as chars_file:
        charset: List[str] = list(sorted(set(chars_file.read())))

    # Clear cache
    if os.path.exists(args.io.glyphs):
        shutil.rmtree(args.io.glyphs)

    # Create new directory for glyphs
    glyphdir = pathlib.Path(args.io.glyphs)
    glyphdir.mkdir(parents=True, exist_ok=True)

    # Load font to use for glyphs and use given size
    font_ascii: FreeTypeFont = ImageFont.truetype(args.io.font, size=args.ascii.size)

    # Compute max width and max height across distinct glyphs
    bboxes: List[Tuple[float, float, float, float]] = [font_ascii.getbbox(text=ch) for ch in charset]
    sizes: List[Tuple[float, ...]] = [(bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes]
    max_w = int(max(w for w, _ in sizes) + 2 * args.ascii.pad)
    max_h = int(max(h for _, h in sizes) + 2 * args.ascii.pad)

    # Generate glyphs
    for index, (char, bbox) in enumerate(zip(charset, bboxes)):
        # Compute per-glyph required size
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        offset_x = (max_w - w) // 2 - bbox[0]
        offset_y = (max_h - h) // 2 - bbox[1]

        # Generate glyphs of character
        image = PIL.Image.new("L", size=[max_w, max_h], color=0)
        draw = ImageDraw.Draw(image)
        draw.text(xy=(offset_x, offset_y), text=char, fill=255, font=font_ascii)
        image.save(glyphdir / f"glyph_{index}.jpg")

    # Save glyph order as a file
    with open(glyphdir / "glyphs.txt", mode="w", encoding="utf-8") as glyphs_file:
        glyphs_file.write("".join(charset))

    # Read glyphs back in memory
    glyphs = []
    for file in filter(lambda path: path.suffix.lower() == ".jpg", sorted(glyphdir.iterdir(), key=lambda path: path.name)):
        glyph = iio.imread(file)
        glyph = torch.from_numpy(glyph)
        glyphs.append(glyph)
    glyphs = torch.stack(glyphs, dim=0)
    glyphs = TF.to_dtype(glyphs, dtype=torch.float32, scale=True)
    glyphs = glyphs.to(device)

    # Retrieve input dimensions
    c_i, h_inp, w_inp = image_input.size()
    n_g, h_gly, w_gly = glyphs.size()

    # Resize input to match glyphgrid
    h_new = h_gly * (s_h := h_inp // h_gly)
    w_new = w_gly * (s_w := w_inp // w_gly)
    image_input = TF.resize(image_input, size=[h_new, w_new]).squeeze(0).to(device)

    # Convert Image to ASCII
    configuration = ASCIIConfig(sizes=(n_g, s_h, s_w), chars=charset, glyphs=glyphs)
    algorithm: Algorithm = SGDAlgorithm(**asdict(args.alg), ascii=configuration, device=args.env.device)
    ascii: ASCIIOutput = algorithm.convert(image_input)

    # Transfer to host
    axes: List[Axes] = plt.subplots(nrows=2, ncols=1)[1]

    # Plot Comparison
    axes[0].grid(False)
    axes[0].set_title("Input")
    axes[0].imshow(image_raw)
    axes[1].grid(False)
    axes[1].set_title("ASCII")
    axes[1].imshow(ascii["image"], cmap="grey")
    plt.tight_layout()
    plt.show()

    # Output filepaths
    input_path = pathlib.Path(args.io.input)
    ascii_image_path = os.path.join(args.io.output, input_path.name)
    ascii_texts_path = os.path.join(args.io.output, input_path.stem + ".txt")

    # Save to disk as image
    TF.to_pil_image(ascii["image"]).save(ascii_image_path)

    # Save to disk as ASCII
    with open(ascii_texts_path, "w", encoding="utf-8") as ascii_file:
        ascii_file.write("\n".join(["".join(line) for line in ascii["chars"]]))


tyro.cli(main)
