import os
import pathlib
import shutil

from typing import List, Literal, Tuple

import cv2 as cv
import einx
import imageio.v3 as iio
import matplotlib.pyplot as plt
import msgspec
import numpy as np
import numpy.typing as npt
import PIL
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
import tyro

from matplotlib.axes import Axes
from PIL import ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.tv_tensors import Image
from tqdm import tqdm
from tyro.conf import OmitArgPrefixes


class Args(msgspec.Struct):
    """Convert an RGB Image to ASCII art"""

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

    # Size of the font text
    size: int = 8

    # Character padding all sides
    pad: float = 0.25

    # Maximum optimization iterations
    max_steps: int = 64

    # Error threshold to end optimization
    eps: float = 0.01

    # Convergence rate
    eta: float = 0.75

    # Gumbel-Softmax smoothness factor
    tau: float = 0.75

    # Seed to reproduce output
    seed: int = 42

    # Accelerator used to run the code
    device: Literal["cpu", "cuda:0", "cuda:1"] = "cuda:0"


def main(args: OmitArgPrefixes[Args]) -> None:
    # Environment
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Read input image as grayscale and preprocess
    image_raw: npt.NDArray = iio.imread(args.input, pilmode="RGB").astype(np.uint8)
    image_grey = cv.cvtColor(image_raw, cv.COLOR_RGB2GRAY)
    image_proc = cv.medianBlur(image_grey, 5)

    # Convert to Tensor
    image_rgb: Image = TF.to_image(image_proc)
    image_input = TF.to_dtype(image_rgb, scale=True)
    image_input = TF.to_grayscale(image_input)

    # Use charset from disk
    with open(args.chars, "r", encoding="utf-8") as chars_file:
        charset: List[str] = list(sorted(set(chars_file.read())))

    # Clear cache
    if os.path.exists(args.glyphs):
        shutil.rmtree(args.glyphs)

    # Create new directory for glyphs
    glyphdir = pathlib.Path(args.glyphs)
    glyphdir.mkdir(parents=True, exist_ok=True)

    # Load font to use for glyphs and use given size
    font_ascii: FreeTypeFont = ImageFont.truetype(args.font, size=args.size)

    # Compute max width and max height across distinct glyphs
    bboxes: List[Tuple[float, float, float, float]] = [font_ascii.getbbox(text=ch) for ch in charset]
    sizes: List[Tuple[float, ...]] = [(bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes]
    max_w = int(max(w for w, _ in sizes) + 2 * args.pad)
    max_h = int(max(h for _, h in sizes) + 2 * args.pad)

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

    # Initialize state
    state_param: Tensor = torch.ones([n_g, s_h, s_w], requires_grad=True, device=device)
    min_state: Tensor = state_param
    solution: Tensor = state_param
    pb = tqdm()
    lr: float = args.eta
    min_loss = 100.0
    step = 0

    # Use stochastic gradient descent algorithm to minimize loss w.r.t. logits matrix
    opt = Adam([state_param], lr=args.eta)
    sch = ReduceLROnPlateau(opt, factor=0.85, patience=25, cooldown=5)

    # Optimization Loop
    while True:
        # Use Gumbel-Softmax to learn soft categorical distribution
        weights: Tensor = F.gumbel_softmax(state_param, tau=args.tau, hard=True, dim=0)
        predict: Tensor = einx.dot("n_g s_h s_w, n_g h_g w_g -> (s_h h_g) (s_w w_g)", weights, glyphs)
        loss: Tensor = F.mse_loss(predict, image_input)

        # Log
        pb.set_description(f"loss={loss.item():.2f}, lr={lr:.2f}")

        # End iteration early
        if loss < args.eps:
            break
        if step > args.max_steps:
            break

        # Save minimum state
        if min_loss > loss:
            min_state = state_param.cpu().clone().detach()  # type: ignore
            solution = predict.cpu().clone().detach()  # type: ignore
            min_loss = loss

        # Adjust learning rate
        if lr != sch.get_last_lr()[0]:
            lr = sch.get_last_lr()[0]
            pb.set_description(f"loss={loss.item():.2f}, lr={lr:.2f}")

        # Propagate Loss
        opt.zero_grad()
        loss.backward()

        # Optimize
        opt.step()
        sch.step(loss.item())

        # Increment
        pb.update()
        step += 1

    # Transfer to host
    axes: List[Axes] = plt.subplots(nrows=2, ncols=1)[1]

    # Plot Comparison
    axes[0].grid(False)
    axes[0].set_title("Input")
    axes[0].imshow(image_raw)
    axes[1].grid(False)
    axes[1].set_title("ASCII")
    axes[1].imshow(solution, cmap="grey")
    plt.tight_layout()
    plt.show()

    # Output filepaths
    input_path = pathlib.Path(args.input)
    ascii_image_path = os.path.join(args.output, input_path.name)
    ascii_texts_path = os.path.join(args.output, input_path.stem + ".txt")

    # Save to disk as image
    TF.to_pil_image(solution).save(ascii_image_path)

    # Save to disk as ASCII
    with open(ascii_texts_path, "w", encoding="utf-8") as ascii_file:
        state_i: List[List[int]] = min_state.argmax(dim=0).tolist()
        ascii: List[List[str]] = [[charset[i] for i in sub] for sub in state_i]
        ascii_file.write("\n".join(["".join(line) for line in ascii]))


tyro.cli(main)
