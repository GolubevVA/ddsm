import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm as tqdm_bar

# Optional system stats (falls back gracefully if unavailable)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# Selene & external modules used by the original repo
from selene_sdk.utils import NonStrandSpecific
from selene_sdk.targets import Target

# Make repo modules importable (matches original script assumptions)
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "external"))

from ddsm import *  # noqa: F401,F403
from sei import *   # noqa: F401,F403
from selene_utils import *  # noqa: F401,F403


# -----------------------------
# Config and utilities
# -----------------------------

@dataclass
class ModelParameters:
    seifeatures_file: str = '../data/target.sei.names'
    seimodel_file: str = '../data/best.sei.model.pth.tar'

    ref_file: str = '../data/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
    ref_file_mmap: str = '../data/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap'
    tsses_file: str = '../data/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'

    fantom_files: Tuple[str, str] = (
        "../data/agg.plus.bw.bedgraph.bw",
        "../data/agg.minus.bw.bedgraph.bw",
    )
    fantom_blacklist_files: Tuple[str, str] = (
        "../data/fantom.blacklist8.plus.bed.gz",
        "../data/fantom.blacklist8.minus.bed.gz",
    )

    diffusion_weights_file: str = 'steps400.cat4.speed_balance.time4.0.samples100000.pth'

    device: str = 'cuda'
    batch_size: int = 64          # safer default for 16GB GPU than original 256
    num_workers: int = 4

    n_time_steps: int = 400
    random_order: bool = False
    speed_balanced: bool = True
    ncat: int = 4

    num_epochs: int = 200
    lr: float = 5e-4

    # Sampling config for "SEI validation metric"
    val_sei_every: int = 1
    val_sei_num_steps: int = 100
    val_sei_batch_size: int = 64
    val_sei_time_dilation: float = 1.0
    val_sei_max_time: float = 4.0
    val_sei_min_time: float = 4.0 / 400.0
    val_sei_k_samples: int = 1  # how many generated samples per validation example for the SEI MSE metric


def _get_git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _system_stats(device: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    # CPU / RAM
    if psutil is not None:
        vm = psutil.virtual_memory()
        stats["sys/ram_used_gb"] = float((vm.total - vm.available) / (1024 ** 3))
        stats["sys/ram_total_gb"] = float(vm.total / (1024 ** 3))
        stats["sys/cpu_percent"] = float(psutil.cpu_percent(interval=None))
    # GPU
    if device.startswith("cuda") and torch.cuda.is_available():
        stats["gpu/mem_allocated_gb"] = float(torch.cuda.memory_allocated() / (1024 ** 3))
        stats["gpu/mem_reserved_gb"] = float(torch.cuda.memory_reserved() / (1024 ** 3))
        try:
            stats["gpu/max_mem_allocated_gb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        except Exception:
            pass
    return stats


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


# -----------------------------
# Data + model definitions (from original script)
# -----------------------------

class GenomicSignalFeatures(Target):
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = {feat: i for i, feat in enumerate(features)}
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        import tabix
        import pyBigWig

        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0): int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                        self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
                            replacement_indices,
                            np.fmax(int(s) - start, 0): int(e) - start
                        ] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat


class TSSDatasetS(Dataset):
    def __init__(self, config: ModelParameters, seqlength=1024, split="train", n_tsses=100000, rand_offset=0):
        self.genome = MemmapGenome(
            input_path=config.ref_file,
            memmapfile=config.ref_file_mmap,
            blacklist_regions='hg38'
        )
        self.tfeature = GenomicSignalFeatures(
            list(config.fantom_files),
            ['cage_plus', 'cage_minus'],
            (2000,),
            list(config.fantom_blacklist_files)
        )

        import pandas as pd
        self.tsses = pd.read_table(config.tsses_file, sep='\t').iloc[:n_tsses, :]

        self.split = split
        if split == "train":
            self.tsses = self.tsses.iloc[~np.isin(self.tsses['chr'].values, ['chr8', 'chr9', 'chr10'])]
        elif split == "valid":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr10'])]
        elif split == "test":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr8', 'chr9'])]
        else:
            raise ValueError(f"Unknown split={split}")

        self.rand_offset = rand_offset
        self.seqlength = seqlength

    def __len__(self):
        return self.tsses.shape[0]

    def __getitem__(self, tssi):
        chrm = self.tsses['chr'].values[tssi]
        pos = self.tsses['TSS'].values[tssi]
        strand = self.tsses['strand'].values[tssi]
        offset = 1 if strand == '-' else 0
        offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)

        seq = self.genome.get_encoding_from_coords(
            chrm,
            pos - int(self.seqlength / 2) + offset,
            pos + int(self.seqlength / 2) + offset,
            strand
        )
        signal = self.tfeature.get_feature_data(
            chrm,
            pos - int(self.seqlength / 2) + offset,
            pos + int(self.seqlength / 2) + offset
        )
        if strand == '-':
            signal = signal[::-1, ::-1]
        return np.concatenate([seq, signal.T], axis=-1).astype(np.float32)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class ScoreNet(nn.Module):
    def __init__(self, embed_dim=256, time_dependent_weights=None, time_step=0.01):
        super().__init__()
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        n = 256
        self.linear = nn.Conv1d(5, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
        ])
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        self.act = lambda x: x * torch.sigmoid(x)
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, 4, kernel_size=1))
        self.register_buffer("time_dependent_weights", time_dependent_weights)
        self.time_step = time_step

    def forward(self, x, t, t_ind=None, return_a=False):
        embed = self.act(self.embed(t / 2))
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            out = h + out if h.shape == out.shape else h
        out = self.final(out).permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]
        out = out - out.mean(axis=-1, keepdims=True)
        return out


# -----------------------------
# Core math: loss + time weights
# -----------------------------

def compute_time_dependent_weights(
    config: ModelParameters,
    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints,
    alpha, beta,
    cache_path: Optional[Path],
) -> torch.Tensor:
    """
    Computes time-dependent weights as in the original script.
    Optionally saves to cache_path.
    """
    if cache_path is not None and cache_path.exists():
        print(f"[stage] loading cached time-dependent weights: {cache_path}", flush=True)
        tdw = torch.load(str(cache_path), map_location="cpu")
        return tdw

    print("[stage] computing time-dependent weights (this can take a while)...", flush=True)
    sb = UnitStickBreakingTransform()

    # Note: use a smaller dataset for this stage to keep it reasonable.
    train_set = TSSDatasetS(config, n_tsses=40000, rand_offset=10)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    time_dependent_cums = torch.zeros(config.n_time_steps).to(config.device)
    time_dependent_counts = torch.zeros(config.n_time_steps).to(config.device)

    for x in tqdm_bar(loader, desc="time-dependent weights", dynamic_ncols=True):
        x = x[..., :4]
        random_t = torch.randint(0, config.n_time_steps, (x.shape[0],))

        if config.random_order:
            order = np.random.permutation(np.arange(config.ncat))
            perturbed_x, perturbed_x_grad = diffusion_factory(
                x[..., order], random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta
            )
            perturbed_x = perturbed_x[..., np.argsort(order)]
            perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
        else:
            perturbed_x, perturbed_x_grad = diffusion_factory(
                x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta
            )

        perturbed_x = perturbed_x.to(config.device)
        perturbed_x_grad = perturbed_x_grad.to(config.device)
        random_t = random_t.to(config.device)

        time_dependent_counts[random_t] += 1

        if config.speed_balanced:
            s_w = 2 / (torch.ones(config.ncat - 1, device=config.device)
                       + torch.arange(config.ncat - 1, 0, -1, device=config.device).float())
        else:
            s_w = torch.ones(config.ncat - 1, device=config.device)

        if config.random_order:
            order = np.random.permutation(np.arange(config.ncat))
            perturbed_v = sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
            td = (perturbed_v * (1 - perturbed_v) * s_w[(None,) * (x.ndim - 1)] *
                  (gx_to_gv(perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2).view(x.shape[0], -1).mean(dim=1).detach()
            time_dependent_cums[random_t] += td
        else:
            perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
            td = (perturbed_v * (1 - perturbed_v) * s_w[(None,) * (x.ndim - 1)] *
                  (gx_to_gv(perturbed_x_grad, perturbed_x)) ** 2).view(x.shape[0], -1).mean(dim=1).detach()
            time_dependent_cums[random_t] += td

    tdw = time_dependent_cums / torch.clamp(time_dependent_counts, min=1)
    tdw = tdw / tdw.mean()

    if cache_path is not None:
        _ensure_dir(cache_path.parent)
        torch.save(tdw.detach().cpu(), str(cache_path))
        print(f"[stage] saved time-dependent weights cache: {cache_path}", flush=True)

    return tdw.detach().cpu()


def ddsm_loss(
    config: ModelParameters,
    sb: UnitStickBreakingTransform,
    score_model: nn.Module,
    x: torch.Tensor,   # [B, L, 4]
    s_signal: torch.Tensor,  # [B, L, 1]
    random_t: torch.LongTensor,  # [B]
    timepoints: torch.Tensor,
    time_dependent_weights: torch.Tensor,  # [T] on CPU ok; we index and move
    v_one, v_zero, v_one_loggrad, v_zero_loggrad,
    alpha, beta,
    create_graph: bool = True
) -> torch.Tensor:
    """
    Computes the per-batch training loss used in the original script.
    """
    # Perturbation
    if config.random_order:
        order = np.random.permutation(np.arange(config.ncat))
        perturbed_x, perturbed_x_grad = diffusion_factory(
            x[..., order].cpu(), random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta
        )
        perturbed_x = perturbed_x[..., np.argsort(order)]
        perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
    else:
        # original used diffusion_fast_flatdirichlet in one branch; diffusion_factory is consistent
        perturbed_x, perturbed_x_grad = diffusion_factory(
            x.cpu(), random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta
        )

    device = config.device
    perturbed_x = perturbed_x.to(device)
    perturbed_x_grad = perturbed_x_grad.to(device)
    s_signal = s_signal.to(device)

    random_timepoints = timepoints[random_t].to(device)
    score = score_model(torch.cat([perturbed_x, s_signal], -1), random_timepoints)

    if config.speed_balanced:
        s_w = 2 / (torch.ones(config.ncat - 1, device=device) + torch.arange(config.ncat - 1, 0, -1, device=device).float())
    else:
        s_w = torch.ones(config.ncat - 1, device=device)

    tdw_sqrt = torch.sqrt(time_dependent_weights.to(device))
    w_t = (1.0 / tdw_sqrt)[random_t][(...,) + (None,) * (x.ndim - 1)]

    if config.random_order:
        order = np.random.permutation(np.arange(config.ncat))
        perturbed_v = sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()
        loss = torch.mean(torch.mean(
            w_t * s_w[(None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) *
            (gx_to_gv(score[..., order], perturbed_x[..., order], create_graph=create_graph)
             - gx_to_gv(perturbed_x_grad[..., order], perturbed_x[..., order])) ** 2,
            dim=(1)
        ))
    else:
        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
        loss = torch.mean(torch.mean(
            w_t * s_w[(None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) *
            (gx_to_gv(score, perturbed_x, create_graph=create_graph)
             - gx_to_gv(perturbed_x_grad, perturbed_x)) ** 2,
            dim=(1)
        ))

    return loss


# -----------------------------
# SEI metric evaluation (sampling)
# -----------------------------

@torch.no_grad()
def compute_sei_h3k4me3(
    sei_model: nn.Module,
    seifeatures_df,
    seqs_oh: np.ndarray,  # [N, 1024, 4] float / int
    batch_size: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    """
    Returns per-sequence mean H3K4me3 score (shape [N]).
    Matches the evaluation style used in eval_promoter_designer.ipynb.
    """
    idx_h3k4 = (seifeatures_df[1].str.strip().values == "H3K4me3")
    out = np.zeros((seqs_oh.shape[0], 21907), dtype=np.float32)

    for i in tqdm_bar(range(int(np.ceil(seqs_oh.shape[0] / batch_size))), desc="SEI forward", dynamic_ncols=True):
        sl = slice(i * batch_size, min((i + 1) * batch_size, seqs_oh.shape[0]))
        seq = seqs_oh[sl]
        # pad to 4096 with 0.25 context as in original scripts
        x = torch.cat([
            torch.ones((seq.shape[0], 4, 1536), device=device) * 0.25,
            torch.tensor(seq, device=device, dtype=torch.float32).transpose(1, 2),
            torch.ones((seq.shape[0], 4, 1536), device=device) * 0.25
        ], dim=2)
        pred = sei_model(x).detach().float().cpu().numpy()
        out[sl] = pred

    return out[:, idx_h3k4].mean(axis=1)


@torch.no_grad()
def sample_sequences(
    score_model: nn.Module,
    sampler_fn,
    concat_input: torch.Tensor,  # [B, 1024, 1]
    batch_size: int,
    device: str,
    max_time: float,
    min_time: float,
    time_dilation: float,
    num_steps: int,
    eps: float,
    speed_balanced: bool,
) -> np.ndarray:
    samples = sampler_fn(
        score_model,
        (1024, 4),
        batch_size=batch_size,
        max_time=max_time,
        min_time=min_time,
        time_dilation=time_dilation,
        num_steps=num_steps,
        eps=eps,
        speed_balanced=speed_balanced,
        device=device,
        concat_input=concat_input.to(device),
    ).detach().cpu().numpy()
    return samples


# -----------------------------
# W&B init
# -----------------------------

def init_wandb(args: argparse.Namespace, config: ModelParameters, run_dir: Path) -> Optional[Any]:
    if args.wandb_mode == "disabled":
        print("[wandb] disabled", flush=True)
        return None

    import wandb

    wandb_kwargs = dict(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=str(run_dir),
    )

    run = wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None})
    # Log config
    cfg = asdict(config)
    cfg.update({
        "cli_args": vars(args),
        "git_commit": _get_git_commit_hash(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
    })
    wandb.config.update(cfg, allow_val_change=True)

    return run


# -----------------------------
# Main train
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train promoter designer score model with W&B logging.")
    # W&B
    p.add_argument("--wandb-entity", dest="wandb_entity", default=None)
    p.add_argument("--wandb-project", dest="wandb_project", default=None)
    p.add_argument("--wandb-name", dest="wandb_name", default=None)
    p.add_argument("--wandb-tags", dest="wandb_tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", dest="wandb_mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="W&B mode. Use 'disabled' to skip all wandb calls.")
    # Output
    p.add_argument("--outdir", type=str, default="runs/promoter_designer", help="Base output directory.")
    p.add_argument("--run-id", type=str, default=None, help="Run id suffix. If omitted, uses timestamp.")
    p.add_argument("--save-every-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    # Training overrides
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)

    # Data/model paths overrides
    p.add_argument("--diffusion-weights", type=str, default=None)
    p.add_argument("--sei-model", type=str, default=None)
    p.add_argument("--sei-features", type=str, default=None)
    p.add_argument("--ref-fa", type=str, default=None)
    p.add_argument("--ref-mmap", type=str, default=None)
    p.add_argument("--tsses", type=str, default=None)

    # Validation behavior
    p.add_argument("--val-every-epochs", type=int, default=1, help="Compute validation loss every N epochs.")
    p.add_argument("--val-sei-every-epochs", type=int, default=1, help="Compute SEI sampling metric every N epochs.")
    p.add_argument("--val-sei-k-samples", type=int, default=1, help="Number of generated samples per example for SEI metric.")
    p.add_argument("--val-sei-subset", type=int, default=2915, help="How many validation examples to use for SEI metric.")
    p.add_argument("--val-sei-num-steps", type=int, default=100)
    p.add_argument("--val-sei-time-dilation", type=float, default=1.0)
    p.add_argument("--val-sei-batch-size", type=int, default=64)

    # Cache
    p.add_argument("--time-weights-cache", type=str, default="runs/cache/time_dependent_weights.pth")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelParameters()

    # Apply CLI overrides
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.num_workers = int(args.num_workers)
    if args.epochs is not None:
        config.num_epochs = int(args.epochs)
    if args.lr is not None:
        config.lr = float(args.lr)
    if args.device is not None:
        config.device = args.device

    if args.diffusion_weights is not None:
        config.diffusion_weights_file = args.diffusion_weights
    if args.sei_model is not None:
        config.seimodel_file = args.sei_model
    if args.sei_features is not None:
        config.seifeatures_file = args.sei_features
    if args.ref_fa is not None:
        config.ref_file = args.ref_fa
    if args.ref_mmap is not None:
        config.ref_file_mmap = args.ref_mmap
    if args.tsses is not None:
        config.tsses_file = args.tsses

    config.val_sei_every = int(args.val_sei_every_epochs)
    config.val_sei_k_samples = int(args.val_sei_k_samples)
    config.val_sei_num_steps = int(args.val_sei_num_steps)
    config.val_sei_time_dilation = float(args.val_sei_time_dilation)
    config.val_sei_batch_size = int(args.val_sei_batch_size)

    run_id = args.run_id or _now_str()
    run_dir = Path(args.outdir) / run_id
    ckpt_dir = run_dir / "checkpoints"
    _ensure_dir(ckpt_dir)

    print(f"[run] output dir: {run_dir}", flush=True)
    print(f"[run] device: {config.device}", flush=True)
    print(f"[run] batch_size: {config.batch_size} | num_workers: {config.num_workers}", flush=True)
    print(f"[run] command: {' '.join(sys.argv)}", flush=True)

    _seed_everything(args.seed)

    # Save static run metadata locally
    meta = {
        "timestamp": time.time(),
        "run_id": run_id,
        "command": " ".join(sys.argv),
        "git_commit": _get_git_commit_hash(),
        "config": asdict(config),
        "cli_args": vars(args),
    }
    _save_json(run_dir / "run_meta.json", meta)

    # W&B
    run = init_wandb(args, config, run_dir)

    # -----------------------------
    # Load SEI model and features
    # -----------------------------
    import pandas as pd

    print("[stage] loading SEI features...", flush=True)
    seifeatures = pd.read_csv(config.seifeatures_file, sep='|', header=None)

    print("[stage] loading SEI model weights...", flush=True)
    sei = nn.DataParallel(NonStrandSpecific(Sei(4096, 21907)))
    sei.load_state_dict(torch.load(config.seimodel_file, map_location='cpu')['state_dict'])
    sei = sei.to(config.device)
    sei.eval()

    # -----------------------------
    # Load diffusion weights
    # -----------------------------
    print(f"[stage] loading diffusion weights: {config.diffusion_weights_file}", flush=True)
    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(config.diffusion_weights_file)
    v_one = v_one.cpu()
    v_zero = v_zero.cpu()
    v_one_loggrad = v_one_loggrad.cpu()
    v_zero_loggrad = v_zero_loggrad.cpu()
    timepoints = timepoints.cpu()

    alpha = torch.ones(config.ncat - 1).float()
    beta = torch.arange(config.ncat - 1, 0, -1).float()

    # -----------------------------
    # Time-dependent weights (cacheable)
    # -----------------------------
    cache_path = Path(args.time_weights_cache) if args.time_weights_cache else None
    time_dependent_weights = compute_time_dependent_weights(
        config, v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints, alpha, beta, cache_path
    )
    # Save plot (nice to have)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(torch.sqrt(time_dependent_weights).numpy())
        plt.title("sqrt(time_dependent_weights)")
        plt.tight_layout()
        plt.savefig(str(run_dir / "timedependent_weight.png"))
        plt.close()
        print(f"[stage] saved {run_dir / 'timedependent_weight.png'}", flush=True)
    except Exception as e:
        print(f"[warn] could not save timedependent_weight.png: {e}", flush=True)

    # -----------------------------
    # Datasets
    # -----------------------------
    print("[stage] preparing datasets...", flush=True)
    train_set = TSSDatasetS(config, split="train", n_tsses=40000, rand_offset=100)
    valid_set = TSSDatasetS(config, split="valid", n_tsses=40000, rand_offset=0)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # For SEI metric: pre-load a fixed validation subset into memory as in original code
    valid_batches = []
    val_subset = int(args.val_sei_subset)
    with torch.no_grad():
        for xb in tqdm_bar(valid_loader, desc="cache valid subset", dynamic_ncols=True):
            valid_batches.append(xb)
            if sum(b.shape[0] for b in valid_batches) >= val_subset:
                break
    valid_batches = valid_batches[:]
    valid_all = np.concatenate([b.numpy() for b in valid_batches], axis=0)[:val_subset]
    valid_concat_input = torch.tensor(valid_all[:, :, 4:5], dtype=torch.float32)
    valid_seqs = valid_all[:, :, :4]

    print("[stage] computing SEI(H3K4me3) for validation reference sequences...", flush=True)
    valid_ref_h3k4 = compute_sei_h3k4me3(
        sei, seifeatures, valid_seqs, batch_size=128, device=config.device
    )
    _save_json(run_dir / "valid_ref_h3k4me3_stats.json", {
        "n": int(valid_ref_h3k4.shape[0]),
        "mean": float(np.mean(valid_ref_h3k4)),
        "std": float(np.std(valid_ref_h3k4)),
    })

    # -----------------------------
    # Model + optimizer
    # -----------------------------
    print("[stage] building score model...", flush=True)
    sb = UnitStickBreakingTransform()
    score_model = nn.DataParallel(ScoreNet(time_dependent_weights=torch.sqrt(time_dependent_weights)))
    score_model = score_model.to(config.device)
    score_model.train()

    optimizer = Adam(score_model.parameters(), lr=config.lr)
    sampler_fn = Euler_Maruyama_sampler

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_sei = float("inf")
    global_step = 0

    for epoch in range(config.num_epochs):
        print(f"epoch {epoch} train:", flush=True)

        score_model.train()
        running = 0.0
        n_items = 0
        t0 = time.time()

        pbar = tqdm_bar(train_loader, dynamic_ncols=True)
        for xS in pbar:
            x = xS[:, :, :4].float()
            s_sig = xS[:, :, 4:5].float()

            # Importance sampling for t (matches original)
            tdw_sqrt = torch.sqrt(time_dependent_weights)
            p = (tdw_sqrt / tdw_sqrt.sum()).cpu().numpy()
            random_t = torch.LongTensor(np.random.choice(np.arange(config.n_time_steps), size=x.shape[0], p=p))

            loss = ddsm_loss(
                config, sb, score_model, x, s_sig, random_t, timepoints,
                time_dependent_weights, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(x.shape[0])
            running += float(loss.item()) * bs
            n_items += bs
            global_step += 1

            # per-step logging
            train_loss = running / max(n_items, 1)
            pbar.set_description(f"loss={train_loss:.6f}")

            if run is not None:
                import wandb
                log = {
                    "train/loss_step": float(loss.item()),
                    "train/loss_epoch_running": float(train_loss),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "epoch": epoch,
                    "step": global_step,
                }
                log.update(_system_stats(config.device))
                wandb.log(log, step=global_step)

        epoch_train_loss = running / max(n_items, 1)
        print(f"[train] epoch {epoch} avg_loss={epoch_train_loss:.6f} time={time.time()-t0:.1f}s", flush=True)

        # -----------------------------
        # Validation loss (per batch, every epoch as requested)
        # -----------------------------
        if (epoch % int(args.val_every_epochs)) == 0:
            print(f"epoch {epoch} validation:", flush=True)
            score_model.eval()
            v_running = 0.0
            v_items = 0
            vpbar = tqdm_bar(valid_loader, dynamic_ncols=True)
            for xS in vpbar:
                x = xS[:, :, :4].float()
                s_sig = xS[:, :, 4:5].float()

                tdw_sqrt = torch.sqrt(time_dependent_weights)
                p = (tdw_sqrt / tdw_sqrt.sum()).cpu().numpy()
                random_t = torch.LongTensor(np.random.choice(np.arange(config.n_time_steps), size=x.shape[0], p=p))

                vloss = ddsm_loss(
                    config, sb, score_model, x, s_sig, random_t, timepoints,
                    time_dependent_weights, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta,
                    create_graph=False,
                )

                bs = int(x.shape[0])
                v_running += float(vloss.item()) * bs
                v_items += bs
                v_avg = v_running / max(v_items, 1)
                vpbar.set_description(f"val_loss={v_avg:.6f}")

                global_step += 1  # keep W&B step strictly increasing
                if run is not None:
                    import wandb
                    log = {
                        "val/loss_step": float(vloss.item()),
                        "val/loss_epoch_running": float(v_avg),
                        "epoch": epoch,
                        "step": global_step,
                    }
                    log.update(_system_stats(config.device))
                    wandb.log(log, step=global_step)

            epoch_val_loss = v_running / max(v_items, 1)
            print(f"[val] epoch {epoch} avg_loss={epoch_val_loss:.6f}", flush=True)

        # -----------------------------
        # SEI validation metric via sampling (expensive; default every epoch on subset)
        # -----------------------------
        if (epoch % config.val_sei_every) == 0:
            print(f"[stage] epoch {epoch}: sampling sequences for SEI metric...", flush=True)
            score_model.eval()

            k = int(config.val_sei_k_samples)
            mses = []
            for k_i in range(k):
                # sample in mini-batches to avoid OOM
                all_gen = []
                for i in tqdm_bar(range(int(np.ceil(valid_concat_input.shape[0] / config.val_sei_batch_size))),
                              desc=f"sample k={k_i+1}/{k}", dynamic_ncols=True):
                    sl = slice(i * config.val_sei_batch_size,
                               min((i + 1) * config.val_sei_batch_size, valid_concat_input.shape[0]))
                    gen = sample_sequences(
                        score_model, sampler_fn,
                        concat_input=valid_concat_input[sl],
                        batch_size=int(sl.stop - sl.start),
                        device=config.device,
                        max_time=config.val_sei_max_time,
                        min_time=config.val_sei_min_time,
                        time_dilation=config.val_sei_time_dilation,
                        num_steps=int(config.val_sei_num_steps),
                        eps=1e-5,
                        speed_balanced=config.speed_balanced,
                    )
                    all_gen.append(gen)
                all_gen = np.concatenate(all_gen, axis=0)
                bin_gen = (all_gen > 0.5).astype(np.float32)
                gen_h3k4 = compute_sei_h3k4me3(sei, seifeatures, bin_gen, batch_size=128, device=config.device)
                mse = float(np.mean((valid_ref_h3k4 - gen_h3k4) ** 2))
                mses.append(mse)

            sei_mse_mean = float(np.mean(mses))
            sei_mse_std = float(np.std(mses))
            print(f"[val-sei] epoch {epoch} H3K4me3_MSE mean={sei_mse_mean:.6f} std={sei_mse_std:.6f}", flush=True)

            if run is not None:
                import wandb
                wandb.log({
                    "val/sei_h3k4me3_mse_mean": sei_mse_mean,
                    "val/sei_h3k4me3_mse_std": sei_mse_std,
                    "epoch": epoch,
                    "step": global_step,
                }, step=global_step)

            # Best checkpoint by SEI metric
            if sei_mse_mean < best_val_sei:
                best_val_sei = sei_mse_mean
                best_path = ckpt_dir / "best_sei.pth"
                ckpt = {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": score_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "git_commit": _get_git_commit_hash(),
                    "command": " ".join(sys.argv),
                    "best_val_sei": best_val_sei,
                }
                torch.save(ckpt, str(best_path))
                print(f"[ckpt] new best SEI metric -> saved: {best_path}", flush=True)
                if run is not None:
                    import wandb
                    wandb.save(str(best_path))

        # -----------------------------
        # Regular checkpoint
        # -----------------------------
        if (epoch % int(args.save_every_epochs)) == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
            ckpt = {
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": score_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "git_commit": _get_git_commit_hash(),
                "command": " ".join(sys.argv),
            }
            torch.save(ckpt, str(ckpt_path))
            print(f"[ckpt] saved: {ckpt_path}", flush=True)
            if run is not None:
                import wandb
                wandb.save(str(ckpt_path))

    print("[done] training finished.", flush=True)
    if run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
