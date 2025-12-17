"""Utility functions"""

import random
import logging
from pathlib import Path

import torch
from torch.optim import Optimizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_model_params(model, trainable_only: bool = True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    checkpoint_dir: str | Path,
    model_state_dict,
    adamw_state_dict,
    step: int,
):
    directory = Path(checkpoint_dir)
    if not directory.exists():
        logger.error(f"Checkpoint directory: {checkpoint_dir}, does not exist!")
        raise FileNotFoundError(
            f"Checkpoint directory: {checkpoint_dir} does not exist!"
        )
    model_checkpoint_path = directory / f"cfm_toy_model_{step}.pt"
    adamw_checkpoint_path = directory / f"cfm_toy_adamw_{step}.pt"
    torch.save(model_state_dict, model_checkpoint_path)
    logger.info(f"Model checkpoint saved at: {model_checkpoint_path}")
    torch.save(adamw_state_dict, adamw_checkpoint_path)
    logger.info(f"adamw checkpoint saved at: {adamw_checkpoint_path}")


def load_checkpoint(
    model,
    model_path: str | Path,
    device: str,
    adamw: Optimizer | None = None,
    adamw_path: str | Path | None = None,
):
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict, strict=True, assign=True)
    if isinstance(adamw, Optimizer) and adamw_path is not None:
        adamw_state_dict = torch.load(adamw_path, map_location=device)
        adamw.load_state_dict(adamw_state_dict, strict=True, assign=True)

    return model, adamw
