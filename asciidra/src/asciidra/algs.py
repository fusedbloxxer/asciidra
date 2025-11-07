from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, NoReturn, Tuple, TypedDict

import einx
import torch
import torch.nn.functional as F

from msgspec import Struct
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .args import DeviceType


class ASCIIOutput(TypedDict):
    chars: List[List[str]]
    image: Tensor


class ASCIIConfig(Struct):
    sizes: Tuple[int, int, int]
    chars: List[str]
    glyphs: Tensor


@dataclass
class Algorithm(ABC):
    ascii: ASCIIConfig

    def __post_init__(self) -> None:
        self.N_G: int = self.ascii.sizes[0]
        self.S_H: int = self.ascii.sizes[1]
        self.S_W: int = self.ascii.sizes[2]

    @abstractmethod
    def convert(self, image: Tensor) -> ASCIIOutput:
        raise NotImplementedError()


@dataclass
class SGDAlgorithm(Algorithm):
    ema: float
    eta: float
    tau: float
    eps: float
    steps: int
    device: DeviceType

    def convert(self, image: Tensor) -> ASCIIOutput:
        # Initialize state
        state_param: Tensor = torch.ones([self.N_G, self.S_H, self.S_W], requires_grad=True, device=self.device)
        min_state: Tensor = state_param
        min_loss: float = 100.0
        ema: float = self.ema
        lr: float = self.eta

        # Use stochastic gradient descent algorithm to minimize loss w.r.t. logits matrix
        opt = Adam([state_param], lr=self.eta)
        sch = ReduceLROnPlateau(opt, factor=0.85, patience=50, cooldown=25)
        pb: tqdm[NoReturn] = tqdm()
        step = 0

        # Optimization Loop
        while True:
            # Use Gumbel-Softmax to learn soft categorical distribution
            weights: Tensor = F.gumbel_softmax(state_param, tau=self.tau, hard=True, dim=0)
            predict: Tensor = einx.dot("n_g s_h s_w, n_g h_g w_g -> (s_h h_g) (s_w w_g)", weights, self.ascii.glyphs)
            predict = F.avg_pool2d(predict[None, None, ...], kernel_size=5, stride=1, padding=2).squeeze()

            # Compute Loss
            loss: Tensor = F.mse_loss(predict, image)

            # Log
            pb.set_description(f"loss={loss.item():.2f}, lr={lr:.2f}")

            # End iteration early
            if loss < self.eps:
                break
            if step > self.steps:
                break

            # Save minimum state
            if min_loss >= loss:
                min_state = ema * min_state + (1 - ema) * state_param
                min_loss = loss.item()

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

        # Stitch ASCII patches
        ascii_probs = min_state.argmax(dim=0)
        ascii_index = F.one_hot(ascii_probs, num_classes=self.N_G).permute((2, 0, 1)).type(torch.float32)
        ascii_image = einx.dot("n_g s_h s_w, n_g h_g w_g -> (s_h h_g) (s_w w_g)", ascii_index, self.ascii.glyphs)
        ascii_image = ascii_image.cpu()
        ascii_text: List[List[str]] = [[self.ascii.chars[i] for i in sub] for sub in ascii_probs.tolist()]
        return {"chars": ascii_text, "image": ascii_image}


@dataclass
class CNNAlgorithm(Algorithm):
    pass


@dataclass
class AEAlgorithm(Algorithm):
    pass
