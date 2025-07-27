import os
import glob
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file
from diffusion_model import flatten_state_dict, get_target_model_flat_dim


class WeightTrajectoryDataset(Dataset):
    def __init__(self, trajectory_dir: str, mode: str = "sequential"):
        self.trajectory_dir = trajectory_dir
        self.mode = mode
        self.weight_files = sorted(
            glob.glob(os.path.join(trajectory_dir, "epoch_*.safetensors")),
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
        if not self.weight_files:
            raise FileNotFoundError(f"No weights found in {trajectory_dir}")

        sample_state = load_file(self.weight_files[0])
        self.flat_dim = get_target_model_flat_dim(sample_state)

        if mode == "sequential":
            self.pairs = [(i, i + 1) for i in range(len(self.weight_files) - 1)]
        else:
            self.pairs = [
                (i, j)
                for i in range(len(self.weight_files))
                for j in range(i + 1, len(self.weight_files))
            ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        w_start = flatten_state_dict(load_file(self.weight_files[i])).float()
        w_end = flatten_state_dict(load_file(self.weight_files[j])).float()
        t_start = torch.tensor([float(i)], dtype=torch.float32)
        t_end = torch.tensor([float(j)], dtype=torch.float32)
        return w_start, w_end, t_start, t_end
