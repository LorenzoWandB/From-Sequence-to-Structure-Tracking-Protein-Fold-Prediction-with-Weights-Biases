#!/usr/bin/env python3
"""
ProteusAI — Bayesian Sweep over ESMFold Fine-tuning (HuggingFace)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import EsmForProteinFolding, AutoTokenizer

import wandb

# ── W&B Configuration (set these to match your setup) ──
ENTITY = "your-wandb-entity"   # your W&B team or username
PROJECT = "your-project-name"  # your W&B project name
DATA_DIR = Path("data")

from train_esm import load_manifest, get_split, output_to_pdb, compute_rmsd, compute_tm_score, kabsch_align_torch

SWEEP_CONFIG = {
    "method": "bayes",
    "name": "esmfold-finetune-sweep",
    "metric": {"name": "val/rmsd", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},
        "weight_decay": {"values": [0.0, 0.01, 0.1]},
        "num_recycles": {"values": [1, 2, 3, 4]},
        "unfreeze_layers": {"values": [0, 2, 4, 6]},
        "max_seq_len": {"values": [256, 384, 512]},
        "epochs": {"value": 6},
    },
}


def sweep_trial():
    run = wandb.init(entity=ENTITY, project=PROJECT, tags=["sweep", "esmfold", "finetune"])
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = load_manifest()
    train_proteins = get_split(manifest, "train")
    val_proteins = get_split(manifest, "val")

    print(f"Loading ESMFold (lr={config.lr:.1e}, unfreeze={config.unfreeze_layers}, recycles={config.num_recycles})")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
    model.config.num_recycles = config.num_recycles

    # Freeze
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "esm_s_combine" in name or "trunk" in name or "distogram_head" in name:
            p.requires_grad = True
        if config.unfreeze_layers > 0:
            for li in range(33 - config.unfreeze_layers, 33):
                if f"layers.{li}." in name:
                    p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"model/trainable_params": trainable})

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=config.lr, weight_decay=config.weight_decay)

    max_len = config.max_seq_len
    best_rmsd = float("inf")

    for epoch in range(config.epochs):
        model.train()
        losses = []
        for protein in train_proteins:
            gene = protein["gene"]
            seq = protein["sequence"][:max_len]
            ref_path = DATA_DIR / "structures" / f"{gene}_coords.npy"
            if not ref_path.exists():
                continue
            ref = torch.tensor(np.load(ref_path)[:max_len], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            try:
                with torch.amp.autocast("cuda"):
                    inputs = tokenizer([seq], return_tensors="pt", padding=False, add_special_tokens=False).to(device)
                    output = model(**inputs)
                    pred = output.positions[-1, 0, :, 1, :]
                    n = min(len(pred), len(ref))
                    pred_aligned = kabsch_align_torch(pred[:n], ref[:n])
                    loss = F.smooth_l1_loss(pred_aligned, ref[:n])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            except RuntimeError:
                torch.cuda.empty_cache()
                continue

        # Val
        model.eval()
        rmsds, tms = [], []
        with torch.no_grad():
            for protein in val_proteins:
                gene = protein["gene"]
                seq = protein["sequence"][:max_len]
                ref_path = DATA_DIR / "structures" / f"{gene}_coords.npy"
                if not ref_path.exists():
                    continue
                ref = np.load(ref_path)[:max_len]
                try:
                    inputs = tokenizer([seq], return_tensors="pt", padding=False, add_special_tokens=False).to(device)
                    output = model(**inputs)
                    _, pred_coords, _ = output_to_pdb(output, seq)
                    n = min(len(pred_coords), len(ref))
                    if n > 10:
                        rmsds.append(compute_rmsd(pred_coords[:n], ref[:n]))
                        tms.append(compute_tm_score(pred_coords[:n], ref[:n]))
                except RuntimeError:
                    torch.cuda.empty_cache()

        avg_rmsd = np.mean(rmsds) if rmsds else float("inf")
        wandb.log({"epoch": epoch+1, "train/loss": np.mean(losses) if losses else 0,
                    "val/rmsd": avg_rmsd, "val/tm_score": np.mean(tms) if tms else 0})
        best_rmsd = min(best_rmsd, avg_rmsd)

    wandb.log({"val/best_rmsd": best_rmsd})
    wandb.finish()
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--sweep-id", type=str, default=None)
    args = parser.parse_args()

    sweep_id = args.sweep_id or wandb.sweep(SWEEP_CONFIG, entity=ENTITY, project=PROJECT)
    print(f"Sweep: {sweep_id} — running {args.count} trials")
    wandb.agent(sweep_id, function=sweep_trial, count=args.count, entity=ENTITY, project=PROJECT)
    print(f"\nhttps://wandb.ai/{ENTITY}/{PROJECT}/sweeps/{sweep_id}")

if __name__ == "__main__":
    main()
