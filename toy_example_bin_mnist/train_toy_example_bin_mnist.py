import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from ddsm import *

from utils import get_best_device

def _setup_wandb_env(args) -> None:
    """CLI overrides env; do not log secrets to wandb config."""
    if getattr(args, "wandb_api_key", None):
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    if getattr(args, "wandb_entity", None):
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    if getattr(args, "wandb_project", None):
        os.environ["WANDB_PROJECT"] = args.wandb_project

    if getattr(args, "wandb_mode", None):
        os.environ["WANDB_MODE"] = args.wandb_mode


# ------------------------
# Data: binarized MNIST
# ------------------------
def load_mnist_binarized(root: str):
    import joblib

    root = Path(root)
    datapath = root / "bin-mnist"
    datapath.mkdir(parents=True, exist_ok=True)
    dataset_path = datapath / "mnist.pkl.gz"

    if not dataset_path.exists():
        datafiles = {
            "train": (
                "http://www.cs.toronto.edu/~larocheh/public/"
                "datasets/binarized_mnist/binarized_mnist_train.amat"
            ),
            "valid": (
                "http://www.cs.toronto.edu/~larocheh/public/"
                "datasets/binarized_mnist/binarized_mnist_valid.amat"
            ),
            "test": (
                "http://www.cs.toronto.edu/~larocheh/public/"
                "datasets/binarized_mnist/binarized_mnist_test.amat"
            ),
        }
        datasplits = {}
        for split, url in datafiles.items():
            print(f"Downloading {split} data...")
            datasplits[split] = np.loadtxt(urlretrieve(url)[0])

        joblib.dump(
            [datasplits["train"], datasplits["valid"], datasplits["test"]],
            open(dataset_path, "wb"),
        )

    x_train, x_valid, x_test = joblib.load(open(dataset_path, "rb"))
    return x_train, x_valid, x_test


class BinMNIST(Dataset):
    """Binary MNIST dataset (returns tensor on CPU)."""

    def __init__(self, data):
        h, w, c = 28, 28, 1
        self.data = torch.tensor(data, dtype=torch.float32).view(-1, c, h, w)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        return self.data[idx]


def get_binmnist_datasets(root: str):
    x_train, x_valid, x_test = load_mnist_binarized(root)
    return (BinMNIST(x_train), BinMNIST(x_valid), BinMNIST(x_test))


def binary_to_onehot(x: torch.Tensor) -> torch.Tensor:
    # x: [B,1,28,28] or [B,28,28]
    x = x.squeeze(1) if x.ndim == 4 else x
    xonehot = [(x == 1)[..., None], (x == 0)[..., None]]
    return torch.cat(xonehot, dim=-1)  # [B,28,28,2]


# ------------------------
# Model: ScoreNet
# ------------------------
class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, channels=(64, 128, 256, 512), embed_dim=512):
        super().__init__()
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = nn.Conv2d(2, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 2, 3, stride=1)

        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        embed = self.act(self.embed(t / 4.0))

        h1 = self.conv1(x.permute(0, 3, 1, 2))
        h1 = self.act(self.gnorm1(h1 + self.dense1(embed)))

        h2 = self.conv2(h1)
        h2 = self.act(self.gnorm2(h2 + self.dense2(embed)))

        h3 = self.conv3(h2)
        h3 = self.act(self.gnorm3(h3 + self.dense3(embed)))

        h4 = self.conv4(h3)
        h4 = self.act(self.gnorm4(h4 + self.dense4(embed)))

        h = self.tconv4(h4)
        h = self.act(self.tgnorm4(h + self.dense5(embed)))

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = self.act(self.tgnorm3(h + self.dense6(embed)))

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = self.act(self.tgnorm2(h + self.dense7(embed)))

        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h.permute(0, 2, 3, 1)
        h = h - h.mean(dim=-1, keepdim=True)
        return h


# ------------------------
# Noise cache
# ------------------------
def ensure_presampled_noise(
    *,
    noise_path: str,
    num_samples: int,
    num_cat: int,
    num_time_steps: int,
    max_time: float,
    order: int,
    steps_per_tick: int,
    mode: str,
    logspace: bool,
    speed_balance: bool,
    device: torch.device,
):
    noise_path = Path(noise_path)
    if noise_path.exists():
        print(f"Using cached presampled noise: {noise_path}")
        return noise_path

    noise_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Presampling noise (will be saved to): {noise_path}")

    torch.set_default_dtype(torch.float32)

    alpha = torch.ones(num_cat - 1, device=device, dtype=torch.float32)
    beta = torch.arange(num_cat - 1, 0, -1, device=device, dtype=torch.float32)

    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = noise_factory(
        num_samples,
        num_time_steps,
        alpha,
        beta,
        total_time=max_time,
        order=order,
        time_steps=steps_per_tick,
        logspace=logspace,
        speed_balanced=speed_balance,
        mode=mode,
        device=device,
    )

    v_one = v_one.cpu()
    v_zero = v_zero.cpu()
    v_one_loggrad = v_one_loggrad.cpu()
    v_zero_loggrad = v_zero_loggrad.cpu()
    timepoints = torch.tensor(timepoints, dtype=torch.float32)  # ensure float32

    torch.save((v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints), noise_path)
    print("Done presampling.")
    return noise_path


# ------------------------
# CLI / Training
# ------------------------
def parse_args():
    p = argparse.ArgumentParser("Train DDSM on binarized MNIST (toy example)")

    # data / io
    p.add_argument("--data_root", type=str, default="./mnist")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--save_name", type=str, default="score_model_binmnist.pth")

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-10)
    p.add_argument("--num_workers", type=int, default=0)

    # model
    p.add_argument("--channels", type=str, default="64,128,256,512")
    p.add_argument("--embed_dim", type=int, default=512)

    # diffusion / sampling params used during training
    p.add_argument("--num_cat", type=int, default=2)
    p.add_argument("--speed_balance", action="store_true", default=False)

    # presampled noise cache (new)
    p.add_argument(
        "--noise_path",
        type=str,
        default="steps400.cat2.time4.0.samples10000.pth",
        help="Path to cached presampled noise .pth. If missing, it will be generated.",
    )
    p.add_argument("--noise_num_samples", type=int, default=10000)
    p.add_argument("--noise_num_time_steps", type=int, default=400)
    p.add_argument("--noise_max_time", type=float, default=4.0)
    p.add_argument("--noise_order", type=int, default=1000)
    p.add_argument("--noise_steps_per_tick", type=int, default=200)
    p.add_argument("--noise_mode", choices=["path", "independent"], default="path")
    p.add_argument("--noise_logspace", action="store_true", default=False)

    # wandb (CLI overrides env; otherwise env is used)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default="DDSM_BinMNIST_toy_example")
    p.add_argument("--wandb_mode", type=str, default=None, help="e.g. online|offline|disabled")
    p.add_argument("--wandb_api_key", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true", default=False)

    # misc
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    torch.set_default_dtype(torch.float32)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = get_best_device()
    print(f"Using device: {device}")

    # parse channels
    channels = tuple(int(x.strip()) for x in args.channels.split(",") if x.strip())

    # ensure presampled noise (cache)
    noise_path = ensure_presampled_noise(
        noise_path=args.noise_path,
        num_samples=args.noise_num_samples,
        num_cat=args.num_cat,
        num_time_steps=args.noise_num_time_steps,
        max_time=args.noise_max_time,
        order=args.noise_order,
        steps_per_tick=args.noise_steps_per_tick,
        mode=args.noise_mode,
        logspace=args.noise_logspace,
        speed_balance=args.speed_balance,
        device=device,
    )

    # Keep presampled noise on CPU (default from torch.load), it's large.
    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(noise_path)

    # alpha/beta for C=2 (keep float32; important for MPS)
    alpha = torch.tensor([1.0], dtype=torch.float32)
    beta = torch.tensor([1.0], dtype=torch.float32)

    # datasets / loaders
    train_set, valid_set, test_set = get_binmnist_datasets(args.data_root)
    weights_est_dl = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    training_dl = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_dl = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # diffusion globals
    C = args.num_cat
    speed_balanced = bool(args.speed_balance)

    if speed_balanced:
        s = 2 / (torch.ones(C - 1, device=device) + torch.arange(C - 1, 0, -1, device=device).float())
    else:
        s = torch.ones(C - 1, device=device)

    sb = UnitStickBreakingTransform()

    # move timepoints to device for later indexing/NN
    timepoints = timepoints.to(device)
    n_time_steps = int(timepoints.shape[0])

    # ------------------------
    # Compute importance sampling weights
    # ------------------------
    time_dependent_cums = torch.zeros(n_time_steps, device=device)
    time_dependent_counts = torch.zeros(n_time_steps, device=device)

    for x in weights_est_dl:
        # Keep x on CPU for diffusion_factory (noise is on CPU)
        x_cpu = binary_to_onehot(x)  # CPU
        random_t_cpu = torch.randint(0, n_time_steps, (x_cpu.shape[0],), device="cpu", dtype=torch.long)
        random_t = random_t_cpu.to(device)  # for indexing device tensors / accumulation

        perturbed_x_cpu, perturbed_x_grad_cpu = diffusion_factory(
            x_cpu, random_t_cpu, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta, device="cpu"
        )

        perturbed_x = perturbed_x_cpu.to(device)
        perturbed_x_grad = perturbed_x_grad_cpu.to(device)

        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
        gvv = gx_to_gv(perturbed_x_grad, perturbed_x, compute_gradlogdet=False)

        time_dependent_cums[random_t] += (
            (perturbed_v * (1 - perturbed_v) * s[(None,) * (x_cpu.ndim - 1)] * (gvv**2))
            .view(x_cpu.shape[0], -1)
            .mean(dim=1)
            .detach()
        )
        time_dependent_counts[random_t] += 1

    time_dependent_weights = time_dependent_cums / time_dependent_counts.clamp(min=1)
    time_dependent_weights = time_dependent_weights / time_dependent_weights.mean().clamp(min=1e-12)

    # ------------------------
    # Model / Optim
    # ------------------------
    score_model = ScoreNet(channels=channels, embed_dim=args.embed_dim).to(device)
    score_model.train()
    optimizer = Adam(score_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ------------------------
    # wandb
    # ------------------------
    run = None
    if not args.no_wandb:
        _setup_wandb_env(args)

        import wandb

        wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "DDSM_testing")
        wandb_mode = args.wandb_mode or os.environ.get("WANDB_MODE")

        init_kwargs = dict(
            entity=wandb_entity,
            project=wandb_project,
            name=args.wandb_name,
            config={
                "model": "ScoreNet (toy) with fourier embedding",
                "channels": channels,
                "embed_dim": args.embed_dim,
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": "Adam",
                "weight_decay": args.weight_decay,
                "noise_path": str(noise_path),
                "noise_num_samples": args.noise_num_samples,
                "noise_num_time_steps": args.noise_num_time_steps,
                "noise_max_time": args.noise_max_time,
                "noise_order": args.noise_order,
                "noise_steps_per_tick": args.noise_steps_per_tick,
                "noise_mode": args.noise_mode,
                "noise_logspace": args.noise_logspace,
                "speed_balance": args.speed_balance,
                "seed": args.seed,
                "device": str(device),
            },
        )
        if wandb_mode:
            init_kwargs["mode"] = wandb_mode

        run = wandb.init(**init_kwargs)
        run.watch(score_model, log_graph=True)

    # loss fn
    def loss_fn(
        score,
        perturbed_x_grad,
        perturbed_x,
        important_sampling_weights=None,
        *,
        create_graph: bool = True,
    ):
        # Use perturbed_x.ndim (do NOT rely on outer-scope `x`)
        nd = perturbed_x.ndim

        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()

        if important_sampling_weights is not None:
            w = 1 / important_sampling_weights[(...,) + (None,) * (nd - 1)]
        else:
            w = 1.0

        return torch.mean(
            torch.mean(
                w
                * s[(None,) * (nd - 1)]
                * perturbed_v
                * (1 - perturbed_v)
                * (
                    gx_to_gv(
                        score,
                        perturbed_x,
                        create_graph=create_graph,
                        compute_gradlogdet=False,
                    )
                    - gx_to_gv(
                        perturbed_x_grad,
                        perturbed_x,
                        create_graph=False,
                        compute_gradlogdet=False,
                    )
                )
                ** 2,
                dim=1,
            )
        )

    # ------------------------
    # Training loop
    # ------------------------
    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        num_items = 0

        for x in training_dl:
            # Keep x on CPU for diffusion_factory
            x_cpu = binary_to_onehot(x)  # CPU

            p = (torch.sqrt(time_dependent_weights) / torch.sqrt(time_dependent_weights).sum()).detach()
            random_t_cpu = torch.from_numpy(
                np.random.choice(np.arange(n_time_steps), size=x_cpu.shape[0], p=p.cpu().numpy())
            ).long()  # CPU
            random_t = random_t_cpu.to(device)  # device copy for indexing timepoints/weights

            perturbed_x_cpu, perturbed_x_grad_cpu = diffusion_factory(
                x_cpu, random_t_cpu, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta, device="cpu"
            )
            perturbed_x = perturbed_x_cpu.to(device)
            perturbed_x_grad = perturbed_x_grad_cpu.to(device)

            random_timepoints = timepoints[random_t]
            score = score_model(perturbed_x, random_timepoints)

            loss = loss_fn(
                score,
                perturbed_x_grad,
                perturbed_x,
                important_sampling_weights=(torch.sqrt(time_dependent_weights))[random_t],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            avg_loss += float(loss.item()) * x_cpu.shape[0]
            num_items += x_cpu.shape[0]

        train_loss = avg_loss / max(1, num_items)
        print(f"Epoch {epoch+1}/{args.num_epochs} | train/loss={train_loss:.6f}")

        # validation for each epoch
        score_model.eval()
        with torch.set_grad_enabled(True):
            vloss_sum = 0.0
            vcount = 0

            for x in valid_dl:
                x_cpu = binary_to_onehot(x)  # CPU

                p = (torch.sqrt(time_dependent_weights) / torch.sqrt(time_dependent_weights).sum()).detach()
                random_t_cpu = torch.from_numpy(
                    np.random.choice(np.arange(n_time_steps), size=x_cpu.shape[0], p=p.cpu().numpy())
                ).long()  # CPU
                random_t = random_t_cpu.to(device)

                # diffusion_factory must run with grad enabled
                perturbed_x_cpu, perturbed_x_grad_cpu = diffusion_factory(
                    x_cpu,
                    random_t_cpu,
                    v_one,
                    v_zero,
                    v_one_loggrad,
                    v_zero_loggrad,
                    alpha,
                    beta,
                    device="cpu",
                )
                perturbed_x = perturbed_x_cpu.to(device)
                perturbed_x_grad = perturbed_x_grad_cpu.to(device)

                random_timepoints = timepoints[random_t]

                # model forward can be no_grad (we don't backprop on val)
                with torch.no_grad():
                    score = score_model(perturbed_x, random_timepoints)

                # loss needs autograd for gx_to_gv (but not create_graph for val)
                loss = loss_fn(
                    score,
                    perturbed_x_grad,
                    perturbed_x,
                    important_sampling_weights=(torch.sqrt(time_dependent_weights))[random_t],
                    create_graph=False,
                )

                vloss_sum += float(loss.item()) * x_cpu.shape[0]
                vcount += x_cpu.shape[0]

            val_loss = vloss_sum / max(1, vcount)
            print(f"Epoch {epoch+1}/{args.num_epochs} | val/loss={val_loss:.6f}")


        score_model.train()

        if run is not None:
            run.log(
                {"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch + 1},
                step=epoch + 1,
                commit=True,
            )

    # ------------------------
    # Save
    # ------------------------
    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / args.save_name
    torch.save(score_model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    if run is not None:
        artifact = wandb.Artifact(
            name="score_model_binmnist",
            type="model",
            description="Trained ScoreNet model on binarized MNIST using DDSM",
        )
        artifact.add_file(str(model_path))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
