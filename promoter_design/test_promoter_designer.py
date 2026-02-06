import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm as tqdm_bar

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

from selene_sdk.utils import NonStrandSpecific
from selene_sdk.targets import Target

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "external"))

from ddsm import *  # noqa: F401,F403
from sei import *   # noqa: F401,F403
from selene_utils import *  # noqa: F401,F403


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
    batch_size: int = 64
    num_workers: int = 0

    n_time_steps: int = 400
    random_order: bool = False
    speed_balanced: bool = True
    ncat: int = 4

    # Sampling defaults
    num_steps: int = 100
    time_dilation: float = 1.0
    max_time: float = 4.0
    min_time: float = 4.0 / 400.0


def _get_git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _system_stats(device: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if psutil is not None:
        vm = psutil.virtual_memory()
        stats["sys/ram_used_gb"] = float((vm.total - vm.available) / (1024 ** 3))
        stats["sys/ram_total_gb"] = float(vm.total / (1024 ** 3))
        stats["sys/cpu_percent"] = float(psutil.cpu_percent(interval=None))
    if device.startswith("cuda") and torch.cuda.is_available():
        stats["gpu/mem_allocated_gb"] = float(torch.cuda.memory_allocated() / (1024 ** 3))
        stats["gpu/mem_reserved_gb"] = float(torch.cuda.memory_reserved() / (1024 ** 3))
    return stats


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
            for blacklist in self.blacklists:
                for _, s, e in blacklist.query(chrom, start, end):
                    wigmat[:, np.fmax(int(s) - start, 0): int(e) - start] = 0

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat


class TSSDatasetS(Dataset):
    def __init__(self, config: ModelParameters, seqlength=1024, split="test", n_tsses=100000, rand_offset=0):
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


@torch.no_grad()
def compute_sei_h3k4me3(
    sei_model: nn.Module,
    seifeatures_df,
    seqs_oh: np.ndarray,
    batch_size: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    idx_h3k4 = (seifeatures_df[1].str.strip().values == "H3K4me3")
    out = np.zeros((seqs_oh.shape[0], 21907), dtype=np.float32)

    for i in tqdm_bar(range(int(np.ceil(seqs_oh.shape[0] / batch_size))), desc="SEI forward", dynamic_ncols=True):
        sl = slice(i * batch_size, min((i + 1) * batch_size, seqs_oh.shape[0]))
        seq = seqs_oh[sl]
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
    concat_input: torch.Tensor,
    batch_size: int,
    device: str,
    max_time: float,
    min_time: float,
    time_dilation: float,
    num_steps: int,
    eps: float,
    speed_balanced: bool,
) -> np.ndarray:
    return sampler_fn(
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


def init_wandb(args: argparse.Namespace, run_dir: Path, cfg: Dict[str, Any]) -> Optional[Any]:
    if args.wandb_mode == "disabled":
        print("[wandb] disabled", flush=True)
        return None
    import wandb
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=str(run_dir),
    )
    wandb.config.update({
        **cfg,
        "cli_args": vars(args),
        "git_commit": _get_git_commit_hash(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
    }, allow_val_change=True)
    return run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate promoter designer checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint from training script.")
    p.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    p.add_argument("--n-examples", type=int, default=2915, help="Number of examples to evaluate (subset for speed).")
    p.add_argument("--k-samples", type=int, default=5, help="Number of generated samples per example (for mean/std MSE).")

    p.add_argument("--batch-size", type=int, default=64, help="Sampling batch size (keep small to avoid OOM).")
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--time-dilation", type=float, default=1.0)
    p.add_argument("--max-time", type=float, default=4.0)
    p.add_argument("--min-time", type=float, default=4.0/400.0)

    p.add_argument("--device", type=str, default="cuda")

    # W&B
    p.add_argument("--wandb-entity", dest="wandb_entity", default=None)
    p.add_argument("--wandb-project", dest="wandb_project", default=None)
    p.add_argument("--wandb-name", dest="wandb_name", default=None)
    p.add_argument("--wandb-tags", dest="wandb_tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", dest="wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])

    p.add_argument("--outdir", type=str, default="runs/promoter_designer_eval")
    p.add_argument("--run-id", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] output dir: {run_dir}", flush=True)
    print(f"[run] checkpoint: {args.checkpoint}", flush=True)
    print(f"[run] split: {args.split} | n_examples: {args.n_examples} | k_samples: {args.k_samples}", flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})

    # Build config (checkpoint config wins, CLI overrides can still change some eval stuff)
    cfg = ModelParameters()
    for k, v in ckpt_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = args.device
    cfg.batch_size = int(args.batch_size)
    cfg.num_steps = int(args.num_steps)
    cfg.time_dilation = float(args.time_dilation)
    cfg.max_time = float(args.max_time)
    cfg.min_time = float(args.min_time)

    run = init_wandb(args, run_dir, {"config": asdict(cfg), "checkpoint_meta": {
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        "best_val_sei": ckpt.get("best_val_sei"),
        "git_commit": ckpt.get("git_commit"),
        "command": ckpt.get("command"),
    }})

    # Load SEI
    import pandas as pd
    print("[stage] loading SEI features + model...", flush=True)
    seifeatures = pd.read_csv(cfg.seifeatures_file, sep='|', header=None)
    sei = nn.DataParallel(NonStrandSpecific(Sei(4096, 21907)))
    sei.load_state_dict(torch.load(cfg.seimodel_file, map_location='cpu')['state_dict'])
    sei = sei.to(cfg.device)
    sei.eval()

    # Time-dependent weights: try to load from checkpoint if present, else skip (ScoreNet uses it as a buffer only)
    tdw = None
    # We stored only model weights; ScoreNet buffer isn't in state_dict. For eval it isn't strictly required for forward.
    # We'll keep the ScoreNet without time-dependent weights; sampling still works.

    # Build score model + load weights
    print("[stage] loading score model from checkpoint...", flush=True)
    score_model = nn.DataParallel(ScoreNet(time_dependent_weights=tdw))
    score_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    score_model = score_model.to(cfg.device)
    score_model.eval()

    sampler_fn = Euler_Maruyama_sampler

    # Load dataset subset
    print("[stage] loading dataset subset...", flush=True)
    ds = TSSDatasetS(cfg, split=args.split, n_tsses=40000, rand_offset=0)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    batches = []
    with torch.no_grad():
        for xb in tqdm_bar(loader, desc="cache eval subset", dynamic_ncols=True):
            batches.append(xb)
            if sum(b.shape[0] for b in batches) >= int(args.n_examples):
                break
    arr = np.concatenate([b.numpy() for b in batches], axis=0)[:int(args.n_examples)]
    concat_input = torch.tensor(arr[:, :, 4:5], dtype=torch.float32)
    ref_seqs = arr[:, :, :4]

    print("[stage] computing SEI(H3K4me3) for reference sequences...", flush=True)
    ref_h3k4 = compute_sei_h3k4me3(sei, seifeatures, ref_seqs, batch_size=128, device=cfg.device)

    # Sample K times and compute MSE distribution
    print("[stage] sampling and computing MSE...", flush=True)
    mses = []
    for k_i in range(int(args.k_samples)):
        all_gen = []
        for i in tqdm_bar(range(int(np.ceil(concat_input.shape[0] / cfg.batch_size))),
                      desc=f"sample k={k_i+1}/{args.k_samples}", dynamic_ncols=True):
            sl = slice(i * cfg.batch_size, min((i + 1) * cfg.batch_size, concat_input.shape[0]))
            gen = sample_sequences(
                score_model, sampler_fn,
                concat_input=concat_input[sl],
                batch_size=int(sl.stop - sl.start),
                device=cfg.device,
                max_time=cfg.max_time,
                min_time=cfg.min_time,
                time_dilation=cfg.time_dilation,
                num_steps=cfg.num_steps,
                eps=1e-5,
                speed_balanced=cfg.speed_balanced,
            )
            all_gen.append(gen)
        all_gen = np.concatenate(all_gen, axis=0)
        bin_gen = (all_gen > 0.5).astype(np.float32)

        gen_h3k4 = compute_sei_h3k4me3(sei, seifeatures, bin_gen, batch_size=128, device=cfg.device)
        mse = float(np.mean((ref_h3k4 - gen_h3k4) ** 2))
        mses.append(mse)
        print(f"[mse] k={k_i+1}: {mse:.6f}", flush=True)

    mse_mean = float(np.mean(mses))
    mse_std = float(np.std(mses))
    print(f"[result] {args.split} SEI(H3K4me3) MSE mean={mse_mean:.6f} std={mse_std:.6f}", flush=True)

    # Save locally
    (run_dir / "results.json").write_text(json.dumps({
        "split": args.split,
        "n_examples": int(args.n_examples),
        "k_samples": int(args.k_samples),
        "mses": mses,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_step": ckpt.get("step"),
    }, indent=2))

    if run is not None:
        import wandb
        wandb.log({
            f"{args.split}/sei_h3k4me3_mse_mean": mse_mean,
            f"{args.split}/sei_h3k4me3_mse_std": mse_std,
            f"{args.split}/sei_h3k4me3_mse_all": mses,
            **_system_stats(cfg.device),
        })
        wandb.save(str(run_dir / "results.json"))
        wandb.finish()


if __name__ == "__main__":
    main()
