"""Model architecture and checkpoint helpers for the EMNIST teaching app."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

import torch
from torch import nn

from qut001.constants import MODEL_WEIGHTS_DIR

DIGIT_NAMES: Final[list[str]] = [str(index) for index in range(10)]
LETTER_NAMES: Final[list[str]] = [chr(ord("A") + index) for index in range(26)]
CLASS_NAMES: Final[list[str]] = [*DIGIT_NAMES, *LETTER_NAMES]
NUM_CLASSES: Final[int] = len(CLASS_NAMES)

# A simple fixed normalization is used by both training and inference. It avoids
# depending on precomputed dataset statistics and maps pixel values to [-1, 1].
INPUT_MEAN: Final[float] = 0.5
INPUT_STD: Final[float] = 0.5

DEFAULT_MODEL_PATH: Final[Path] = Path(__file__).with_name("emnist36_cnn.pt")


class EMNIST36CNN(nn.Module):
    """A compact CNN for digits 0-9 and case-insensitive letters A-Z."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.12),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def configured_model_path() -> Path:
    """Return the checkpoint path, allowing an environment-variable override."""
    return Path(os.path.join(MODEL_WEIGHTS_DIR, 'emnist36_cnn.pt'))

def save_checkpoint(model: EMNIST36CNN, path: Path) -> None:
    """Save weights together with the class mapping and input normalization."""

    checkpoint = {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "input_mean": INPUT_MEAN,
        "input_std": INPUT_STD,
    }
    torch.save(checkpoint, path)


def load_model(device: torch.device | str = "cpu") -> EMNIST36CNN:
    """Load the trained model or raise a clear setup error."""

    model_path = configured_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model was found at {model_path}. "
            "Run `python train_model.py` once, then restart the app."
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        saved_names = checkpoint.get("class_names")
        if saved_names is not None and list(saved_names) != CLASS_NAMES:
            raise RuntimeError(
                "The checkpoint class mapping does not match this app. "
                "Delete it and run `python train_model.py` again."
            )
    else:
        # Also accept a plain state_dict for convenience.
        state_dict = checkpoint

    model = EMNIST36CNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
