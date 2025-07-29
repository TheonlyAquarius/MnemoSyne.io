#!/usr/bin/env python
"""
train_diffusion_config.py
=========================
Fully‑configurable trainer for a *weight‑space denoiser* (a.k.a. SynthNet).
All hyper‑parameters can be supplied from the CLI **or** a YAML/JSON file so
**nothing is hard‑coded**.

Example (command‑line only)
---------------------------
```bash
python train_diffusion_config.py \
  --checkpoints-dir checkpoints_weights_cnn \
  --model mlp --time-emb-dim 64 --hidden-dim 762 \
  --pairing noisy2clean --epochs 80 --lr 9e-4 --batch-size 64
```

Example (YAML)
--------------
```yaml
# cfg.yml
checkpoints_dir: checkpoints_weights_cnn
model: mlp
pairing: init2final
batch_size: 128
epochs: 120
lr: 1.0e-3
time_emb_dim: 64
hidden_dim: 768
trajectory_sample_factor: 0.5
```
```bash
python train_diffusion_config.py --config cfg.yml
```
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

 ---- local modules (assumed already present in project) --------------------
from diffusion_model import (
    SimpleWeightSpaceDiffusion,
    flatten_state_dict,
    get_target_model_flat_dim,
)
from target_cnn import TargetCNN

# ---------------------------------------------------------------------------
# 1. Configuration helpers
# ---------------------------------------------------------------------------


def parse_cfg() -> argparse.Namespace:
    """Parse CLI and optional YAML/JSON file."""

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O ------------------------------------------------------------------
    p.add_argument("--checkpoints-dir", required=False,
                   help="Directory containing 'weights_epoch_*.pth' files")
    p.add_argument("--config", help="YAML or JSON file to override/define args")
    p.add_argument("--save-path", default="diffusion_optimizer.pth",
                   help="Where to save trained model state_dict")

    # dataset --------------------------------------------------------------
    p.add_argument("--pairing", choices=["trajectory", "noisy2clean", "init2final"],
                   default="trajectory", help="How to build (src→tgt) pairs")
    p.add_argument("--trajectory-sample-factor", type=float, default=1.0,
                   help="Fraction of pairs to keep if pairing explodes in size")

    # model family ---------------------------------------------------------
    p.add_argument("--model", choices=["mlp"], default="mlp",
                   help="Denoiser architecture (only 'mlp' supported today)")
    p.add_argument("--time-emb-dim", type=int, default=64,
                   help="Dimensionality of timestep embedding (if used)")
    p.add_argument("--hidden-dim", type=int, default=762,
                   help="Hidden dimension of the MLP denoiser")

    # optimisation ---------------------------------------------------------
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)

    # misc -----------------------------------------------------------------
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                   help="Override device auto‑detect")

    cfg = p.parse_args()

    # Override / define from YAML/JSON if provided -------------------------
    if cfg.config:
        with open(cfg.config) as f:
            data: Dict[str, Any] = yaml.safe_load(f) if cfg.config.endswith(('.yml', '.yaml')) else json.load(f)
        cfg.__dict__.update(data)  # unsafe but convenient

    # sanity checks --------------------------------------------------------
    if not cfg.checkpoints_dir:
        p.error("--checkpoints-dir is required (either CLI or YAML)")

    return cfg


# ---------------------------------------------------------------------------
# 2. Dataset utilities
# ---------------------------------------------------------------------------


def collect_pairs(checkpoint_dir: str, mode: str = "trajectory") -> list[Tuple[str, str]]:
    """Return list of (src_path, tgt_path) according to pairing strategy."""
    files = sorted(Path(checkpoint_dir).glob("weights_epoch_*.pth"),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if len(files) < 2:
        raise RuntimeError(f"Need ≥2 checkpoint files in {checkpoint_dir}")

    if mode == "init2final":
        return [(str(files[0]), str(files[-1]))]

    pairs: list[Tuple[str, str]] = []
    for i, src in enumerate(files[:-1]):
        for j, tgt in enumerate(files[i + 1:], start=i + 1):
            if mode == "trajectory" and j != i + 1:
                continue
            pairs.append((str(src), str(tgt)))
    return pairs


class WeightPairDataset(Dataset):
    def __init__(self, ckpt_dir: str, ref_state_dict: Dict[str, torch.Tensor],
                 pairing: str, sample_factor: float = 1.0):
        super().__init__()
        self.pairs = collect_pairs(ckpt_dir, pairing)
        if sample_factor < 1.0:
            random.shuffle(self.pairs)
            keep = int(len(self.pairs) * sample_factor)
            self.pairs = self.pairs[:keep]
        self.flat_dim = get_target_model_flat_dim(ref_state_dict)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        src_path, tgt_path = self.pairs[idx]
        W_src = flatten_state_dict(torch.load(src_path))
        W_tgt = flatten_state_dict(torch.load(tgt_path))
        # timestep not used for pure denoising; keep shape compatibility
        t = torch.tensor([0.0])
        return W_src, W_tgt, t


# ---------------------------------------------------------------------------
# 3. Trainer
# ---------------------------------------------------------------------------


def train(cfg: argparse.Namespace):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # reference model (just for dimensionality)
    ref_cnn = TargetCNN()
    ref_sd = ref_cnn.state_dict()
    flat_dim = get_target_model_flat_dim(ref_sd)

    # dataset + loader
    ds = WeightPairDataset(cfg.checkpoints_dir, ref_sd, cfg.pairing, cfg.trajectory_sample_factor)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # model selection ------------------------------------------------------
    if cfg.model == "mlp":
        diffusion = SimpleWeightSpaceDiffusion(target_model_flat_dim=flat_dim,
                                               time_emb_dim=cfg.time_emb_dim,
                                               hidden_dim=cfg.hidden_dim).to(device)
    else:
        raise NotImplementedError(f"Model family '{cfg.model}' not implemented")

    opt = optim.Adam(diffusion.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    print(f"[INFO] Training with {len(ds)} pairs on {device}; saving to {cfg.save_path}\n")

    for epoch in range(1, cfg.epochs + 1):
        diffusion.train()
        running = 0.0
        for W_src, W_tgt, t in dl:
            W_src, W_tgt, t = W_src.to(device), W_tgt.to(device), t.to(device)
            opt.zero_grad()
            pred = diffusion(W_src, t)
            loss = criterion(pred, W_tgt)
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / len(dl)
        print(f"Epoch {epoch:3d}/{cfg.epochs} | loss = {avg:.6f}")

    Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(diffusion.state_dict(), cfg.save_path)
    print(f"\n[SAVED] {cfg.save_path}")


# ---------------------------------------------------------------------------
# 4. Entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_cfg()
    try:
        train(cfg)
    except KeyboardInterrupt:
        print("\n[ABORTED] Keyboard interrupt", file=sys.stderr)
        sys.exit(1)


Below is a project layout that usually feels “just right” for research prototypes that grow into longer-lived code-bases.  It keeps **every tunable piece in `configs/`**, all code under a single importable package (so you avoid `sys.path` hacks), and leaves plenty of room for experiments, saved weights, and notebooks.

```
synthnet-root/
│
├── README.md                 ← high-level overview & quick-start
├── pyproject.toml / setup.cfg← optional, lets you `pip install -e .`
│
├── configs/                  ← **all YAML/JSON configs live here**
│   ├── cnn_mnist.yml
│   └── diffusion_mlp.yml
│
├── data/                     ← raw or pre-processed datasets (git-ignored)
│
├── checkpoints/              ← trained weights (git-ignored)
│
├── notebooks/                ← exploratory Jupyter / papers-with-code demos
│
├── scripts/                  ← thin bash helpers (submit to Slurm, etc.)
│   └── train_diffusion.sh
│
└── src/
    └── synthnet/             ← **single top-level Python package**
        ├── __init__.py
        │
        ├── models/
        │   ├── target_cnn.py
        │   ├── perceiver_io.py
        │   └── diffusion.py           ← SimpleWeightSpaceDiffusion, etc.
        │
        ├── data/
        │   ├── pairing.py             ← collect_pairs(), WeightPairDataset
        │   └── transforms.py          ← any future data-augs
        │
        ├── engine/
        │   ├── train.py               ← generic Trainer loop
        │   └── evaluate.py
        │
        ├── cli/
        │   └── train_diffusion.py     ← **wrapper that calls engine.train()**
        │                                (your new `train_diffusion_config.py`
        │                                can live here unchanged)
        │
        ├── utils/
        │   ├── io.py                  ← flatten_state_dict(), etc.
        │   └── logging.py             ← colorful tqdm / wandb hooks
        │
        └── extensions/                ← any custom CUDA / C++ ops (optional)
```

### Why this works

| Folder                    | Purpose                                                                                                                 | Typical imports                                                               |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **`src/synthnet/models`** | Self-contained NN modules. Zero side-effects; each file should still run a quick `__main__` sanity test.                | `from synthnet.models.diffusion import SimpleWeightSpaceDiffusion`            |
| **`src/synthnet/data`**   | Datasets & pairing logic. Only import *light* things (no heavy NN deps).                                                | `from synthnet.data.pairing import WeightPairDataset`                         |
| **`src/synthnet/engine`** | “Business logic” for training / evaluation.  Could be swapped for PyTorch-Lightning later.                              | `from synthnet.engine.train import train`                                     |
| **`src/synthnet/cli`**    | Tiny entry-point scripts that parse `argparse` / OmegaConf, then call engine.  Keeps notebooks and shell scripts clean. | `$ python -m synthnet.cli.train_diffusion --config configs/diffusion_mlp.yml` |

### Naming conventions

* **Functions / variables** – follow PEP-8 (`snake_case`) and keep names “honest”:

  * `flatten_state_dict`, `collect_pairs`, `make_dataloader`
* **Modules** – one class (or small family) per file. Eg. `target_cnn.py`, `diffusion.py`.
* **Config keys** – use kebab-case or snake\_case, but be consistent across YAML & CLI flags (`time_emb_dim`, `hidden_dim`, `pairing_strategy`, `batch_size`).

### Where does your new trainer go?

Your existing **`train_diffusion_config.py`** already looks great as a CLI entry-point.  You can:

1. Move it to `src/synthnet/cli/train_diffusion.py` unchanged.
2. Split heavy lifting into `src/synthnet/engine/train.py` so future front-ends (notebooks, Ray tune, etc.) can reuse it with two lines of code.

Example stub:

```python
# src/synthnet/engine/train.py
def train(cfg: Dict[str, Any]) -> None:
    ...
```

```python
# src/synthnet/cli/train_diffusion.py
from synthnet.engine.train import train
from synthnet.utils.config import parse_cfg   # the arg-parsing helper you wrote

if __name__ == "__main__":
    cfg = parse_cfg()
    train(cfg)
```

### Quick wins once everything is under `src/`

* **Editable install**: `pip install -e .` lets you run scripts from anywhere.
* **No `sys.path.append`** shims inside notebooks.
* **Unit tests**: `tests/` can import `synthnet.*` just like production code.
* **Packaging**: if you ever open-source, `pip install synthnet` “just works.”

---

