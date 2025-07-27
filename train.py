import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import WeightTrajectoryDataset

from utils import (
    get_target_model,
    get_optimizer_model,
    save_model_weights,
    load_model_weights,
)


TRAJ_DIR = "trajectories"
CHECKPOINT_DIR = "checkpoints"


def train_target(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_target_model(args.target_model).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    opt_class = getattr(optim, args.optimizer)
    optimizer = opt_class(model.parameters(), lr=args.lr)

    run_dir = os.path.join(TRAJ_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_model_weights(model, os.path.join(run_dir, "epoch_0.safetensors"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        save_model_weights(model, os.path.join(run_dir, f"epoch_{epoch}.safetensors"))


def train_optimizer(args):
    trajectory_path = os.path.join(TRAJ_DIR, args.run_name)
    dataset = WeightTrajectoryDataset(trajectory_path, mode=args.training_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_optimizer_model(
        args.optimizer_model,
        flat_dim=dataset.flat_dim,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    if args.resume_from:
        load_model_weights(model, args.resume_from)
        start_epoch = int(os.path.basename(args.resume_from).split("_")[-1].split(".")[0]) + 1

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    criterion = nn.MSELoss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        count = 0
        for w_start, w_end, t_start, t_end in dataloader:
            w_start = w_start.to(device)
            w_end = w_end.to(device)
            t_start = t_start.to(device)
            t_end = t_end.to(device)
            opt.zero_grad()
            pred = model(w_start, t_start, t_end)
            loss = criterion(pred, w_end)
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.optimizer_model}_epoch_{epoch}.safetensors")
        save_model_weights(model, ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)

    pt = subparsers.add_parser("train-target")
    pt.add_argument("--run-name", required=True)
    pt.add_argument("--target-model", default="TargetCNN")
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--optimizer", default="Adam")
    pt.add_argument("--lr", type=float, default=1e-3)

    po = subparsers.add_parser("train-optimizer")
    po.add_argument("--run-name", required=True)
    po.add_argument("--optimizer-model", default="PerceiverOptimizer")
    po.add_argument("--epochs", type=int, default=20)
    po.add_argument("--batch-size", type=int, default=32)
    po.add_argument("--lr", type=float, default=1e-3)
    po.add_argument("--training-mode", choices=["sequential", "permutation"], default="sequential")
    po.add_argument("--resume-from", default=None)

    args = parser.parse_args()

    if args.phase == "train-target":
        train_target(args)
    else:
        train_optimizer(args)


if __name__ == "__main__":
    main()
