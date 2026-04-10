#!/usr/bin/env python3
"""
ProteusAI — ESMFold Structure Prediction (HuggingFace Transformers)
====================================================================
Uses HuggingFace's ESMFold implementation — same Meta model, stable deps.

Two modes:
  1. inference: Run pretrained ESMFold on drug targets (sequence → 3D)
  2. finetune: Fine-tune structure module on train split, eval on held-out

Usage:
    uv run python train_esm.py --mode inference
    uv run python train_esm.py --mode finetune
    uv run python train_esm.py --mode finetune --unfreeze-backbone
"""

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import EsmForProteinFolding, AutoTokenizer

import wandb

# ── W&B Configuration ──
# Set these to your W&B entity (team or username) and project name
ENTITY = "your-wandb-entity"   # e.g. "my-team" or your W&B username
PROJECT = "your-project-name"  # e.g. "proteus-fold"
DATA_DIR = Path("data")


# ═══════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════

def load_manifest():
    p = DATA_DIR / "manifest.json"
    if not p.exists():
        raise FileNotFoundError("Run download_data.py first")
    with open(p) as f:
        return json.load(f)

def get_split(manifest, split):
    return [p for p in manifest if p.get("split") == split]


# ═══════════════════════════════════════════════════════════
# Structural Alignment (Kabsch algorithm)
# ═══════════════════════════════════════════════════════════

def kabsch_align(pred, true):
    """Align pred onto true using Kabsch algorithm (rigid superposition).
    Returns aligned pred coordinates."""
    # Center both
    pred_center = pred.mean(axis=0)
    true_center = true.mean(axis=0)
    pred_c = pred - pred_center
    true_c = true - true_center

    # Covariance matrix
    H = pred_c.T @ true_c
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and translation
    pred_aligned = (pred_c @ R.T) + true_center
    return pred_aligned


def kabsch_align_torch(pred, true):
    """Kabsch alignment for PyTorch tensors. Returns aligned pred."""
    # SVD requires float32 — cast up then back
    orig_dtype = pred.dtype
    pred = pred.float()
    true = true.float()

    pred_c = pred - pred.mean(dim=0, keepdim=True)
    true_c = true - true.mean(dim=0, keepdim=True)

    H = pred_c.T @ true_c
    U, S, Vt = torch.linalg.svd(H)

    d = torch.det(Vt.T @ U.T)
    sign_m = torch.eye(3, device=pred.device)
    sign_m[2, 2] = torch.sign(d)

    R = Vt.T @ sign_m @ U.T
    pred_aligned = (pred_c @ R.T) + true.mean(dim=0, keepdim=True)
    return pred_aligned.to(orig_dtype)


# ═══════════════════════════════════════════════════════════
# Metrics (with alignment)
# ═══════════════════════════════════════════════════════════

def extract_ca_coords(pdb_string):
    coords, plddts = [], []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            plddts.append(float(line[60:66]))
    return np.array(coords), np.array(plddts)

def compute_rmsd(pred, true):
    """RMSD after Kabsch alignment."""
    pred_aligned = kabsch_align(pred, true)
    return np.sqrt(((pred_aligned - true) ** 2).sum(-1).mean())

def compute_tm_score(pred, true):
    """TM-score after Kabsch alignment."""
    pred_aligned = kabsch_align(pred, true)
    L = len(pred)
    d0 = max(1.24 * (max(L, 15) - 15) ** (1.0 / 3.0) - 1.8, 0.5)
    di = np.sqrt(((pred_aligned - true) ** 2).sum(-1))
    return (1.0 / (1.0 + (di / d0) ** 2)).mean()

def compute_gdt_ts(pred, true):
    """GDT-TS after Kabsch alignment."""
    pred_aligned = kabsch_align(pred, true)
    d = np.sqrt(((pred_aligned - true) ** 2).sum(-1))
    return sum((d < c).mean() for c in [1.0, 2.0, 4.0, 8.0]) / 4.0


# ═══════════════════════════════════════════════════════════
# PDB conversion (HuggingFace output → PDB string)
# ═══════════════════════════════════════════════════════════

def output_to_pdb(output, sequence):
    """Convert HuggingFace ESMFold output to PDB string with CA coords."""
    positions = output.positions[-1, 0]  # last recycle, batch 0
    # positions shape: [seq_len, atom_types, 3] — atom 1 is CA
    ca_coords = positions[:, 1, :].detach().cpu().numpy()
    plddt = (output.plddt[0, :, 1].detach().cpu().numpy()) * 100

    residues = ["ALA", "GLY", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP",
                "MET", "SER", "THR", "CYS", "TYR", "ASN", "GLN", "ASP",
                "GLU", "LYS", "ARG", "HIS"]
    lines = []
    for i in range(min(len(ca_coords), len(sequence))):
        x, y, z = ca_coords[i]
        res = residues[i % len(residues)]
        bf = min(plddt[i], 99.99) if i < len(plddt) else 0.0
        lines.append(
            f"ATOM  {i+1:5d}  CA  {res} A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bf:6.2f}           C"
        )
    lines.append("END")
    return "\n".join(lines), ca_coords, plddt


# ═══════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════

def make_plddt_fig(plddts, gene, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = []
    for v in plddts:
        if v > 90: colors.append("#0053D6")
        elif v > 70: colors.append("#65CBF3")
        elif v > 50: colors.append("#FFDB13")
        else: colors.append("#FF7D45")
    ax.bar(range(len(plddts)), plddts, color=colors, width=1.0)
    ax.set_xlabel("Residue")
    ax.set_ylabel("pLDDT")
    ax.set_ylim(0, 100)
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=70, color="gray", linestyle=":", alpha=0.3)
    ax.set_title(f"{gene} — pLDDT {title_suffix}")
    plt.tight_layout()
    return fig

def make_comparison_fig(pred_plddts, ref_plddts, gene):
    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(len(pred_plddts), len(ref_plddts))
    ax.plot(range(n), ref_plddts[:n], color="black", alpha=0.5, linewidth=1, label="AlphaFold ref")
    ax.bar(range(n), pred_plddts[:n], alpha=0.6, width=1.0, label="ESMFold")
    ax.set_xlabel("Residue")
    ax.set_ylabel("pLDDT")
    ax.set_ylim(0, 100)
    ax.set_title(f"{gene} — ESMFold vs AlphaFold")
    ax.legend()
    plt.tight_layout()
    return fig

def make_distance_matrix_fig(pred_coords, true_coords, title=""):
    pd = np.sqrt(((pred_coords[:, None] - pred_coords[None, :]) ** 2).sum(-1))
    td = np.sqrt(((true_coords[:, None] - true_coords[None, :]) ** 2).sum(-1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.imshow(pd, cmap="viridis", vmin=0, vmax=40); a1.set_title("ESMFold")
    a2.imshow(td, cmap="viridis", vmin=0, vmax=40); a2.set_title("AlphaFold Ref")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════

def run_inference(args):
    manifest = load_manifest()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading ESMFold from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.eval().to(device)

    max_len = args.max_seq_len
    print(f"Device: {device}, max_seq_len: {max_len}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name=args.run_name or f"esmfold-inference-{datetime.now().strftime('%m%d-%H%M')}",
        config={"mode": "inference", "model": "ESMFold v1 (HuggingFace)", "max_seq_len": max_len,
                "num_targets": len(manifest),
                "infrastructure": {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                                   "cluster": "CoreWeave SUNK (H100)"}},
        tags=["inference", "esmfold", "real-data", "drug-targets"],
    )

    results_table = wandb.Table(columns=[
        "gene", "area", "role", "seq_len", "mean_plddt",
        "plddt_gt90_pct", "plddt_gt70_pct", "rmsd_vs_af", "tm_vs_af",
    ])

    for i, protein in enumerate(manifest):
        gene = protein["gene"]
        seq = protein["sequence"][:max_len]
        area = protein.get("area", "")
        role = protein.get("role", "")

        print(f"\n[{i+1}/{len(manifest)}] {gene} ({len(seq)} aa, {area})...")

        with torch.no_grad():
            try:
                inputs = tokenizer([seq], return_tensors="pt", padding=False, add_special_tokens=False).to(device)
                output = model(**inputs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    shorter = seq[:max_len // 2]
                    print(f"  OOM — retrying with {len(shorter)} aa")
                    inputs = tokenizer([shorter], return_tensors="pt", padding=False, add_special_tokens=False).to(device)
                    output = model(**inputs)
                    seq = shorter
                else:
                    raise

        pdb_str, pred_coords, pred_plddts = output_to_pdb(output, seq)
        mean_plddt = pred_plddts.mean()
        gt90 = (pred_plddts > 90).mean() * 100
        gt70 = (pred_plddts > 70).mean() * 100

        print(f"  pLDDT: mean={mean_plddt:.1f}, >90={gt90:.0f}%, >70={gt70:.0f}%")

        # Save PDB
        pred_path = DATA_DIR / "predictions" / f"{gene}_esmfold.pdb"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(pdb_str)

        # Log 3D structure
        wandb.log({f"structures/{gene}": wandb.Molecule(str(pred_path))})

        # pLDDT profile
        fig = make_plddt_fig(pred_plddts, gene, "(ESMFold)")
        wandb.log({f"plddt_profile/{gene}": wandb.Image(fig)})
        plt.close(fig)

        # Compare with AlphaFold ref
        rmsd_af, tm_af = None, None
        ref_path = DATA_DIR / "structures" / f"{gene}_coords.npy"
        ref_plddt_path = DATA_DIR / "structures" / f"{gene}_plddt.npy"

        if ref_path.exists():
            ref_coords = np.load(ref_path)
            ref_plddts = np.load(ref_plddt_path)
            n = min(len(pred_coords), len(ref_coords))
            if n > 10:
                rmsd_af = compute_rmsd(pred_coords[:n], ref_coords[:n])
                tm_af = compute_tm_score(pred_coords[:n], ref_coords[:n])
                print(f"  vs AlphaFold: RMSD={rmsd_af:.2f}, TM={tm_af:.3f}")

                fig = make_comparison_fig(pred_plddts, ref_plddts, gene)
                wandb.log({f"comparison/{gene}": wandb.Image(fig)})
                plt.close(fig)

                fig = make_distance_matrix_fig(pred_coords[:n], ref_coords[:n], f"{gene}")
                wandb.log({f"distance_matrix/{gene}": wandb.Image(fig)})
                plt.close(fig)

                ref_pdb = DATA_DIR / "structures" / f"{gene}_{protein['uniprot_id']}.pdb"
                if ref_pdb.exists():
                    wandb.log({f"reference/{gene}": wandb.Molecule(str(ref_pdb))})

        results_table.add_data(
            gene, area, role, len(seq), round(mean_plddt, 1),
            round(gt90, 1), round(gt70, 1),
            round(rmsd_af, 2) if rmsd_af else None,
            round(tm_af, 3) if tm_af else None,
        )
        wandb.log({f"metrics/{gene}_plddt": mean_plddt})
        torch.cuda.empty_cache()

    wandb.log({"drug_target_portfolio": results_table})
    wandb.finish()
    print(f"\nDone! https://wandb.ai/{ENTITY}/{PROJECT}")


# ═══════════════════════════════════════════════════════════
# Fine-tuning
# ═══════════════════════════════════════════════════════════

def run_finetune(args):
    manifest = load_manifest()
    train_proteins = get_split(manifest, "train")
    val_proteins = get_split(manifest, "val")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading ESMFold from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)

    # Freeze strategy
    total_params = sum(p.numel() for p in model.parameters())
    if args.unfreeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if "esm_s_combine" in name or "trunk" in name or "distogram_head" in name:
                p.requires_grad = True
            for li in range(33 - args.unfreeze_layers, 33):
                if f"layers.{li}." in name:
                    p.requires_grad = True
        strategy = f"unfreeze-last-{args.unfreeze_layers}"
    else:
        for name, p in model.named_parameters():
            if "esm" in name.lower():
                p.requires_grad = False
            else:
                p.requires_grad = True
        strategy = "freeze-backbone"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params:,} total, {trainable:,} trainable — {strategy}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    config = {
        "mode": "finetune", "model": "ESMFold v1 (HuggingFace)", "strategy": strategy,
        "total_params": total_params, "trainable_params": trainable,
        "train_proteins": [p["gene"] for p in train_proteins],
        "val_proteins": [p["gene"] for p in val_proteins],
        "epochs": args.epochs, "lr": args.lr, "max_seq_len": args.max_seq_len,
        "num_recycles": args.num_recycles,
        "infrastructure": {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                           "cluster": "CoreWeave SUNK (H100)"},
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name=args.run_name or f"esmfold-ft-{strategy}-{datetime.now().strftime('%m%d-%H%M')}",
        config=config,
        tags=["finetune", "esmfold", "real-data", strategy],
    )

    max_len = args.max_seq_len
    model.config.num_recycles = args.num_recycles
    best_val_rmsd = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for protein in train_proteins:
            gene = protein["gene"]
            seq = protein["sequence"][:max_len]
            ref_path = DATA_DIR / "structures" / f"{gene}_coords.npy"
            if not ref_path.exists():
                continue

            ref_coords = torch.tensor(np.load(ref_path)[:max_len], dtype=torch.float32).to(device)
            optimizer.zero_grad()

            try:
                inputs = tokenizer([seq], return_tensors="pt", padding=False,
                                   add_special_tokens=False).to(device)
                with torch.amp.autocast("cuda"):
                    output = model(**inputs)
                # Compute loss in float32 (Kabsch SVD requires it)
                pred_ca = output.positions[-1, 0, :, 1, :].float()
                ref_f32 = ref_coords[:max_len].float()
                n = min(len(pred_ca), len(ref_f32))
                pred_aligned = kabsch_align_torch(pred_ca[:n], ref_f32[:n])
                fape = F.smooth_l1_loss(pred_aligned, ref_f32[:n])
                pred_d = torch.cdist(pred_ca[:n].unsqueeze(0), pred_ca[:n].unsqueeze(0)).squeeze()
                true_d = torch.cdist(ref_f32[:n].unsqueeze(0), ref_f32[:n].unsqueeze(0)).squeeze()
                disto = F.smooth_l1_loss(pred_d, true_d)
                loss = fape + 0.3 * disto

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

                wandb.log({"train/loss": loss.item(), "train/fape": fape.item(),
                           "train/disto": disto.item(), "train/lr": optimizer.param_groups[0]["lr"],
                           "train/epoch": epoch, "train/protein": gene})
            except (RuntimeError, IndexError) as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM on {gene} — skipping")
                else:
                    print(f"  Error on {gene}: {type(e).__name__} — skipping")
                torch.cuda.empty_cache()
                continue

        scheduler.step()
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("inf")

        # Validation
        model.eval()
        val_m = {"rmsd": [], "tm": [], "gdt": [], "plddt": []}

        with torch.no_grad():
            for protein in val_proteins:
                gene = protein["gene"]
                seq = protein["sequence"][:max_len]
                ref_path = DATA_DIR / "structures" / f"{gene}_coords.npy"
                if not ref_path.exists():
                    continue
                ref_coords = np.load(ref_path)[:max_len]

                try:
                    inputs = tokenizer([seq], return_tensors="pt", padding=False,
                                       add_special_tokens=False).to(device)
                    output = model(**inputs)
                    pdb_str, pred_coords, pred_plddts = output_to_pdb(output, seq)
                    n = min(len(pred_coords), len(ref_coords))

                    if n > 10:
                        val_m["rmsd"].append(compute_rmsd(pred_coords[:n], ref_coords[:n]))
                        val_m["tm"].append(compute_tm_score(pred_coords[:n], ref_coords[:n]))
                        val_m["gdt"].append(compute_gdt_ts(pred_coords[:n], ref_coords[:n]))
                        val_m["plddt"].append(pred_plddts.mean())

                    # Log structures periodically
                    if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == args.epochs - 1:
                        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
                            f.write(pdb_str)
                            wandb.log({f"val_structures/{gene}_e{epoch+1}": wandb.Molecule(f.name)})
                            os.unlink(f.name)

                        fig = make_plddt_fig(pred_plddts, gene, f"(Epoch {epoch+1})")
                        wandb.log({f"val_plddt/{gene}": wandb.Image(fig)})
                        plt.close(fig)

                        if n > 10:
                            fig = make_distance_matrix_fig(pred_coords[:n], ref_coords[:n],
                                                           f"{gene} — Epoch {epoch+1}")
                            wandb.log({f"val_distmat/{gene}": wandb.Image(fig)})
                            plt.close(fig)

                except (RuntimeError, IndexError):
                    torch.cuda.empty_cache()
                    continue

        avg = {k: np.mean(v) for k, v in val_m.items() if v}
        print(f"Epoch {epoch+1}/{args.epochs} | Loss={avg_loss:.4f} | "
              f"RMSD={avg.get('rmsd',0):.2f} TM={avg.get('tm',0):.3f} pLDDT={avg.get('plddt',0):.1f}")

        wandb.log({"epoch": epoch+1, "val/loss": avg_loss, "val/rmsd": avg.get("rmsd", 0),
                    "val/tm_score": avg.get("tm", 0), "val/gdt_ts": avg.get("gdt", 0),
                    "val/plddt_mean": avg.get("plddt", 0)})

        # Save best
        rmsd = avg.get("rmsd", float("inf"))
        if rmsd < best_val_rmsd:
            best_val_rmsd = rmsd
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = "checkpoints/best_esmfold.pt"
            torch.save({"epoch": epoch+1, "strategy": strategy, "metrics": avg,
                         "model_state_dict": {k: v for k, v in model.state_dict().items()
                                               if any(p.data_ptr() == v.data_ptr()
                                                      for p in model.parameters() if p.requires_grad)}}, ckpt)
            artifact = wandb.Artifact("esmfold-finetuned", type="model",
                                       metadata={"epoch": epoch+1, "strategy": strategy, **avg})
            artifact.add_file(ckpt)
            run.log_artifact(artifact)
            print(f"  → Best (RMSD={rmsd:.2f})")

    wandb.finish()
    print(f"\nDone! Best RMSD: {best_val_rmsd:.2f}")
    print(f"https://wandb.ai/{ENTITY}/{PROJECT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["inference", "finetune"], default="inference")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-recycles", type=int, default=3)
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--unfreeze-layers", type=int, default=4)
    p.add_argument("--data-dir", type=str, default="data", help="Data directory (data or data_large)")
    args = p.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir)

    if args.mode == "inference":
        run_inference(args)
    else:
        run_finetune(args)

if __name__ == "__main__":
    main()
