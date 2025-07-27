import os
import torch
from safetensors.torch import save_file, load_file
from models import TargetCNN
from models.perceiver_optimizer import PerceiverOptimizer


def get_target_model(name: str, **kwargs):
    if name == "TargetCNN":
        return TargetCNN(**kwargs)
    raise ValueError(f"Unknown target model: {name}")


def get_optimizer_model(name: str, **kwargs):
    if name == "PerceiverOptimizer":
        return PerceiverOptimizer(**kwargs)
    raise ValueError(f"Unknown optimizer model: {name}")


def save_model_weights(model: torch.nn.Module, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_file(model.state_dict(), filepath)


def load_model_weights(model: torch.nn.Module, filepath: str):
    state = load_file(filepath)
    model.load_state_dict(state)
    return model
