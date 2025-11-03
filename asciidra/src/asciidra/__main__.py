import torch
import tyro


def main() -> None:
    print("Hello from asciidra!")
    print(torch.cuda.is_available())


tyro.cli(main)
