# ProteusAI — Protein Structure Prediction with ESMFold + W&B

Real protein structure prediction pipeline for drug discovery, using ESMFold (Meta AI, 700M params) on 30 drug targets and 500 diverse proteins — tracked end-to-end in [Weights & Biases](https://wandb.ai).

**Example W&B Project:** [View Demo Results](https://wandb.ai/Lorenzo-Team/proteus-fold)

## What This Is

A production-grade ML pipeline that:
1. **Predicts 3D protein structures** from amino acid sequences using ESMFold
2. **Fine-tunes** the model on 500 diverse proteins, evaluates on 30 held-out drug targets
3. **Tracks everything** in W&B — 3D structures, training curves, sweeps, model registry

This is what a real biotech company would build to manage their structure prediction workflow.

## W&B Features Showcased

| Feature | What It Does | File |
|---------|-------------|------|
| **`wandb.Molecule`** | Interactive 3D protein structure visualization with pLDDT confidence coloring | `train_esm.py` |
| **`wandb.Image`** | pLDDT profiles, distance matrices, ESMFold vs AlphaFold comparisons | `train_esm.py` |
| **`wandb.Table`** | Drug target portfolio with 30 targets, metrics, therapeutic areas | `train_esm.py` |
| **`wandb.Artifact`** | Versioned model checkpoints with metrics metadata | `train_esm.py` |
| **`wandb.log`** | Real-time training loss, RMSD, TM-score, pLDDT, learning rate | `train_esm.py` |
| **`wandb.sweep`** | Bayesian hyperparameter optimization (15 trials) | `run_sweep.py` |
| **`artifact.link`** | Model registry with staging/production aliases | `run_registry.py` |
| **Config tracking** | Full model config, infrastructure, data provenance | `train_esm.py` |

## Drug Target Portfolio

30 real targets across four therapeutic areas:

| Area | Targets |
|------|---------|
| **Oncology** (15) | TP53, EGFR, BRAF, KRAS, HER2, CDK2, AKT1, PIK3CA, ABL1, SRC, MDM2, BCL2, PTEN, CDK4, AURORA_A |
| **Rare Disease** (5) | GBA1 (Gaucher), CFTR (Cystic Fibrosis), NF1, GLA (Fabry), SMN1 (SMA) |
| **Infectious Disease** (5) | SARS-CoV-2 Spike, HIV-1 RT, HCV NS5B, P.falciparum DHFR, Influenza Neuraminidase |
| **Metabolic** (5) | JAK2, ERK2, RAF1, FGFR2, ALK |

## Training Results

4-phase progression mirroring a real biotech ML workflow:

| Phase | Run | Epochs | Loss | RMSD | TM-Score | pLDDT |
|-------|-----|--------|------|------|----------|-------|
| 2 — Frozen backbone (lr=5e-5) | `v2-frozen-lr5e-5` | 30 | 7.3 → 2.9 | 13.19 | 0.398 | 69.6 |
| 2 — Frozen backbone (lr=1e-4) | `v2-frozen-lr1e-4` | 30 | 7.4 → 2.0 | 13.99 | 0.389 | 74.2 |
| 3 — Unfreeze 4 layers | `v2-unfreeze4` | 30 | 2.0 → 1.7 | 13.57 | 0.398 | 69.8 |
| 4 — Production (3 recycles) | `v2-production` | 40 | 2.0 → 0.6 | 13.55 | 0.390 | 67.4 |

## Quick Start

### Prerequisites
- GPU with ≥40GB VRAM (H100, A100, etc.)
- Python 3.11+
- W&B account ([wandb.ai](https://wandb.ai))

### Configure

Before running, set your W&B entity and project name in these files:

| File | Variables to set |
|------|-----------------|
| `train_esm.py` | `ENTITY`, `PROJECT` |
| `run_sweep.py` | `ENTITY`, `PROJECT` |
| `run_registry.py` | `ENTITY`, `PROJECT`, `REGISTRY_NAME` |
| `slurm_train_v2.sh` | Cluster partition name, repo path |

```python
# In train_esm.py, run_sweep.py, run_registry.py:
ENTITY = "your-wandb-entity"    # your W&B team or username
PROJECT = "your-project-name"   # your W&B project name
```

### Setup
```bash
pip install torch transformers accelerate wandb numpy requests matplotlib
wandb login
```

### Run
```bash
# 1. Download proteins (~5 min)
python download_data.py         # 30 drug targets
python download_data_large.py   # 500 diverse proteins for training

# 2. Baseline inference — predict all 30 drug targets (~5 min on H100)
python train_esm.py --mode inference --max-seq-len 256

# 3. Fine-tune on 500 proteins, evaluate on drug targets (~30 min per run)
python train_esm.py --mode finetune --epochs 30 --lr 1e-4 --max-seq-len 128 --data-dir data_large

# 4. Fine-tune with backbone unfreezing
python train_esm.py --mode finetune --epochs 30 --lr 3e-5 --unfreeze-backbone --unfreeze-layers 4 --data-dir data_large

# 5. Hyperparameter sweep (15 Bayesian trials)
python run_sweep.py --count 15

# 6. Promote best model to W&B Model Registry
python run_registry.py
```

### Slurm (HPC Cluster)
Edit `slurm_train_v2.sh` to match your cluster (partition name, GPU type, repo path), then:
```bash
sbatch slurm_train_v2.sh
```

## File Structure

```
proteus-fold/
├── train_esm.py           # Main training script (inference + fine-tuning + all W&B logging)
├── download_data.py        # Download 30 drug targets from UniProt + AlphaFold DB
├── download_data_large.py  # Download 500 diverse proteins for training
├── run_sweep.py            # W&B Bayesian hyperparameter sweep
├── run_registry.py         # Promote best model to W&B Model Registry
├── slurm_train_v2.sh       # Slurm job script for HPC clusters
└── README.md               # This file
```

## How W&B Is Used (Code Walkthrough)

### 1. Experiment Tracking (`train_esm.py`)

```python
# Initialize with full config
run = wandb.init(
    entity=ENTITY, project=PROJECT,
    config={
        "model": "ESMFold v1 (HuggingFace)",
        "strategy": "freeze-backbone",
        "trainable_params": 689_000_000,
        "train_proteins": ["P04637", "P31749", ...],
        "val_proteins": ["TP53", "KRAS", ...],
        ...
    },
    tags=["finetune", "esmfold", "real-data"],
)
```

### 2. 3D Protein Structure Logging

```python
# Log predicted structure as interactive 3D molecule
wandb.log({"structures/TP53": wandb.Molecule("predicted.pdb")})

# Log AlphaFold reference for comparison
wandb.log({"reference/TP53": wandb.Molecule("alphafold_ref.pdb")})
```

### 3. Rich Visualizations

```python
# pLDDT confidence profile with AlphaFold-standard coloring
fig = make_plddt_fig(plddts, gene)  # blue >90, cyan >70, yellow >50, orange <50
wandb.log({"plddt_profile/TP53": wandb.Image(fig)})

# Side-by-side distance matrices (predicted vs reference)
wandb.log({"distance_matrix/TP53": wandb.Image(distmat_fig)})

# ESMFold vs AlphaFold pLDDT comparison
wandb.log({"comparison/TP53": wandb.Image(comparison_fig)})
```

### 4. Drug Target Portfolio Table

```python
results_table = wandb.Table(columns=[
    "gene", "area", "role", "seq_len", "mean_plddt",
    "plddt_gt90_pct", "plddt_gt70_pct", "rmsd_vs_af", "tm_vs_af",
])
for protein in portfolio:
    results_table.add_data(gene, area, role, ...)
wandb.log({"drug_target_portfolio": results_table})
```

### 5. Training Metrics

```python
# Per-step training metrics
wandb.log({
    "train/loss": loss.item(),
    "train/fape": fape_loss.item(),
    "train/disto": distogram_loss.item(),
    "train/lr": optimizer.param_groups[0]["lr"],
})

# Per-epoch validation on held-out drug targets
wandb.log({
    "val/rmsd": avg_rmsd,       # Kabsch-aligned RMSD
    "val/tm_score": avg_tm,     # Structural similarity
    "val/plddt_mean": avg_plddt, # Confidence score
    "val/gdt_ts": avg_gdt,      # Global distance test
})
```

### 6. Model Artifacts

```python
artifact = wandb.Artifact("esmfold-finetuned", type="model",
    metadata={"epoch": epoch, "rmsd": best_rmsd, "strategy": "unfreeze-4"})
artifact.add_file("checkpoints/best_esmfold.pt")
run.log_artifact(artifact)
```

### 7. Bayesian Hyperparameter Sweep (`run_sweep.py`)

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/rmsd", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},
        "num_recycles": {"values": [1, 2, 3, 4]},
        "unfreeze_layers": {"values": [0, 2, 4, 6]},
    },
}
sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
wandb.agent(sweep_id, function=sweep_trial, count=15)
```

### 8. Model Registry (`run_registry.py`)

```python
# Link best model to registry with production alias
artifact.link(f"{ENTITY}/wandb-registry-model/{REGISTRY_NAME}")
artifact.aliases.append("production")
artifact.aliases.append("best")
artifact.save()
```

## Biology Background

- **pLDDT** (predicted Local Distance Difference Test): Per-residue confidence score 0-100. Above 90 = reliable for drug design.
- **RMSD** (Root Mean Square Deviation): Distance between predicted and true atom positions after structural alignment (Kabsch). Lower is better.
- **TM-score**: Global structural similarity metric. >0.5 means same fold.
- **Kabsch alignment**: Rigid superposition of two structures before comparing — standard in structural biology.
- **ESMFold**: Meta AI's protein language model. Predicts 3D structure from sequence alone (no MSA needed). 700M parameters.

## License

MIT
