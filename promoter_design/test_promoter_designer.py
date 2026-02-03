import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import pyBigWig
import tabix
import wandb
import time
import argparse

import sys
sys.path.append("../")
sys.path.append("../external")

from ddsm import *
from sei import *
from selene_utils import *


class ModelParameters:
    seifeatures_file = '../data/target.sei.names'
    seimodel_file = '../data/best.sei.model.pth.tar'

    ref_file = '../data/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
    ref_file_mmap = '../data/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap'
    tsses_file = '../data/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'

    fantom_files = [
        "../data/agg.plus.bw.bedgraph.bw",
        "../data/agg.minus.bw.bedgraph.bw"
    ]

    fantom_blacklist_files = [
        "../data/fantom.blacklist8.plus.bed.gz",
        "../data/fantom.blacklist8.minus.bed.gz"
    ]

    diffusion_weights_file = 'steps400.cat4.speed_balance.time4.0.samples100000.pth'
    best_model_path = 'sdedna_promoter_revision.sei.bestvalid.pth'

    device = 'cuda'
    batch_size = 256
    num_workers = 4

    n_time_steps = 400
    random_order = False
    speed_balanced = True

    ncat = 4

    # WandB config
    wandb_project = 'promoter-designer'
    wandb_run_name = 'test-eval'


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
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

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
                        self.blacklists, self.blacklists_indices, self.replacement_indices,
                        self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
                                                                                                replacement_indices,
                                                                                                np.fmax(int(s) - start,
                                                                                                        0): int(
                                                                                                    e) - start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat


class TSSDatasetS(Dataset):
    def __init__(self, config, seqlength=1024, split="train", n_tsses=100000, rand_offset=0):
        self.shuffle = False

        self.genome = MemmapGenome(
            input_path=config.ref_file,
            memmapfile=config.ref_file_mmap,
            blacklist_regions='hg38'
        )
        self.tfeature = GenomicSignalFeatures(
            config.fantom_files,
            ['cage_plus', 'cage_minus'],
            (2000,),
            config.fantom_blacklist_files
        )

        self.tsses = pd.read_table(config.tsses_file, sep='\t')
        self.tsses = self.tsses.iloc[:n_tsses, :]

        self.chr_lens = self.genome.get_chr_lens()
        self.split = split
        if split == "train":
            self.tsses = self.tsses.iloc[~np.isin(self.tsses['chr'].values, ['chr8', 'chr9', 'chr10'])]
        elif split == "valid":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr10'])]
        elif split == "test":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr8', 'chr9'])]
        else:
            raise ValueError
        self.rand_offset = rand_offset
        self.seqlength = seqlength

    def __len__(self):
        return self.tsses.shape[0]

    def __getitem__(self, tssi):
        chrm, pos, strand = self.tsses['chr'].values[tssi], self.tsses['TSS'].values[tssi], self.tsses['strand'].values[
            tssi]
        offset = 1 if strand == '-' else 0

        offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)
        seq = self.genome.get_encoding_from_coords(chrm, pos - int(self.seqlength / 2) + offset,
                                                   pos + int(self.seqlength / 2) + offset, strand)

        signal = self.tfeature.get_feature_data(chrm, pos - int(self.seqlength / 2) + offset,
                                                pos + int(self.seqlength / 2) + offset)
        if strand == '-':
            signal = signal[::-1, ::-1]
        return np.concatenate([seq, signal.T], axis=-1).astype(np.float32)

    def reset(self):
        np.random.seed(0)


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
        self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
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
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])

        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
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
            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final(out)
        out = out.permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]

        out = out - out.mean(axis=-1, keepdims=True)
        return out


def get_sei_predictions(sei_model, seqs, batch_size=128):
    with torch.no_grad():
        preds = np.zeros((seqs.shape[0], 21907))
        for i in range(int(seqs.shape[0] / batch_size)):
            seq = seqs[i * batch_size:(i + 1) * batch_size]
            preds[i * batch_size:(i + 1) * batch_size] = sei_model(
                torch.cat([torch.ones((seq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(seq).transpose(1, 2),
                           torch.ones((seq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
        seq = seqs[-batch_size:]
        preds[-batch_size:] = sei_model(
            torch.cat([torch.ones((seq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(seq).transpose(1, 2),
                       torch.ones((seq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
    return preds


def evaluate_model(score_model, test_datasets, sei_model, seifeatures, timepoints, config):
    sampler = Euler_Maruyama_sampler
    min_time = timepoints[0].item()
    max_time = timepoints[-1].item()

    # Generate samples for different time dilations
    allsamples = []
    allsamples2x = []
    allsamples4x = []

    for t in test_datasets:
        samples = []
        samples2x = []
        samples4x = []

        for i in range(5):
            score_model.eval()
            samples.append(sampler(score_model,
                                   (1024, 4),
                                   batch_size=t.shape[0],
                                   max_time=max_time,
                                   min_time=min_time,
                                   time_dilation=1,
                                   num_steps=100,
                                   eps=1e-5,
                                   speed_balanced=config.speed_balanced,
                                   device=config.device,
                                   concat_input=t[:, :, 4:5].cuda()).cpu().detach().numpy())

            samples2x.append(Euler_Maruyama_sampler(score_model,
                                                    (1024, 4),
                                                    batch_size=t.shape[0],
                                                    max_time=max_time,
                                                    min_time=min_time,
                                                    time_dilation=2,
                                                    time_dilation_start_time=1,
                                                    num_steps=200,
                                                    eps=1e-5,
                                                    speed_balanced=config.speed_balanced,
                                                    device=config.device,
                                                    concat_input=t[:, :, 4:5].cuda()).cpu().detach().numpy())

            samples4x.append(Euler_Maruyama_sampler(score_model,
                                                    (1024, 4),
                                                    batch_size=t.shape[0],
                                                    max_time=max_time,
                                                    min_time=min_time,
                                                    time_dilation=4,
                                                    time_dilation_start_time=1,
                                                    num_steps=400,
                                                    eps=1e-5,
                                                    speed_balanced=config.speed_balanced,
                                                    device=config.device,
                                                    concat_input=t[:, :, 4:5].cuda()).cpu().detach().numpy())

        allsamples.append(samples)
        allsamples2x.append(samples2x)
        allsamples4x.append(samples4x)

    # Convert to numpy arrays
    allsamples = np.concatenate(allsamples, axis=1)
    allsamples2x = np.concatenate(allsamples2x, axis=1)
    allsamples4x = np.concatenate(allsamples4x, axis=1)

    # Get test sequences
    testseqs = np.concatenate(test_datasets, axis=0)[:, :, :4]

    # Get SEI predictions
    testseqs_pred = get_sei_predictions(sei_model, testseqs)
    testseqs_predh3k4me3 = testseqs_pred[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=1)

    # Get samples predictions
    allsamples_pred = np.zeros((5, 2915, 21907))
    allsamples_pred2x = np.zeros((5, 2915, 21907))
    allsamples_pred4x = np.zeros((5, 2915, 21907))

    for j in range(5):
        allsamples_pred[j] = get_sei_predictions(sei_model, allsamples[j] > 0.5)
        allsamples_pred2x[j] = get_sei_predictions(sei_model, allsamples2x[j] > 0.5)
        allsamples_pred4x[j] = get_sei_predictions(sei_model, allsamples4x[j] > 0.5)

    # Calculate metrics
    allsamples_predh3k4me3 = allsamples_pred[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=-1)
    allsamples2x_predh3k4me3 = allsamples_pred2x[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=-1)
    allsamples4x_predh3k4me3 = allsamples_pred4x[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=-1)

    # Calculate MSE for different dilations
    mse = np.mean([((allsamples_predh3k4me3[i] - testseqs_predh3k4me3) ** 2).mean() for i in range(5)])
    mse2x = np.mean([((allsamples2x_predh3k4me3[i] - testseqs_predh3k4me3) ** 2).mean() for i in range(5)])
    mse4x = np.mean([((allsamples4x_predh3k4me3[i] - testseqs_predh3k4me3) ** 2).mean() for i in range(5)])

    return {
        'mse': mse,
        'mse2x': mse2x,
        'mse4x': mse4x
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test promoter designer with wandb logging")
    parser.add_argument("--wandb_entity", default=None, help="Weights & Biases entity")
    parser.add_argument("--wandb_project", default=None, help="Weights & Biases project")
    parser.add_argument("--wandb_run_name", default=None, help="Weights & Biases run name")
    args = parser.parse_args()

    config = ModelParameters()
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        entity=args.wandb_entity,
        name=config.wandb_run_name,
        config={
            'batch_size': config.batch_size,
            'n_time_steps': config.n_time_steps,
            'random_order': config.random_order,
            'speed_balanced': config.speed_balanced,
            'ncat': config.ncat,
            'best_model_path': config.best_model_path
        }
    )

    # Load SEI model
    seifeatures = pd.read_csv(config.seifeatures_file, sep='|', header=None)
    sei = nn.DataParallel(NonStrandSpecific(Sei(4096, 21907)))
    sei.load_state_dict(torch.load(config.seimodel_file, map_location='cpu', weights_only=False)['state_dict'])
    sei.cuda()

    # Load diffusion weights
    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(config.diffusion_weights_file, weights_only=False)
    v_one = v_one.cpu()
    v_zero = v_zero.cpu()
    v_one_loggrad = v_one_loggrad.cpu()
    v_zero_loggrad = v_zero_loggrad.cpu()
    timepoints = timepoints.cpu()

    # Load best model
    score_model = nn.DataParallel(ScoreNet(time_dependent_weights=None))
    score_model.load_state_dict(torch.load(config.best_model_path, map_location='cpu', weights_only=False))
    score_model.cuda()
    score_model.eval()

    # Load test dataset
    test_set = TSSDatasetS(config, split='test', n_tsses=40000, rand_offset=0)
    test_data_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_datasets = []
    for x in test_data_loader:
        test_datasets.append(x)

    # Evaluate
    metrics = evaluate_model(score_model, test_datasets, sei, seifeatures, timepoints, config)

    # Log metrics
    wandb.log({
        'test/mse': metrics['mse'],
        'test/mse2x': metrics['mse2x'],
        'test/mse4x': metrics['mse4x']
    })

    print("Test evaluation completed")
    print(metrics)

    wandb.finish()
