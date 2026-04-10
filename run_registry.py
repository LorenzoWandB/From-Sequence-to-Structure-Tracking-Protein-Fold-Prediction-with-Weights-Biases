#!/usr/bin/env python3
"""
ProteusAI — Model Registry Promotion
======================================
After training, promote best checkpoints through the W&B Model Registry
with staging → production aliases.

Usage:
    uv run python run_registry.py
"""

import wandb

# ── W&B Configuration (set these to match your setup) ──
ENTITY = "your-wandb-entity"         # your W&B team or username
PROJECT = "your-project-name"        # your W&B project name
REGISTRY_NAME = "your-model-name"    # name in W&B Model Registry


def main():
    api = wandb.Api()

    # Find all model artifacts from this project
    print(f"Scanning {ENTITY}/{PROJECT} for model artifacts...")
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": {"$in": ["esm2"]}})

    best_runs = []
    for run in runs:
        if run.state != "finished":
            continue
        rmsd = run.summary.get("val/rmsd") or run.summary.get("val/best_rmsd")
        tm = run.summary.get("val/tm_score", 0)
        plddt = run.summary.get("val/plddt_mean", 0)
        if rmsd is not None:
            best_runs.append({
                "run_id": run.id,
                "name": run.name,
                "rmsd": rmsd,
                "tm_score": tm,
                "plddt": plddt,
                "tags": run.tags,
            })

    if not best_runs:
        print("No finished runs with metrics found. Train first!")
        return

    # Sort by RMSD (lower is better)
    best_runs.sort(key=lambda r: r["rmsd"])

    print(f"\nFound {len(best_runs)} runs with metrics:")
    for i, r in enumerate(best_runs[:10]):
        marker = " ← best" if i == 0 else ""
        print(f"  {r['name']:40s} RMSD={r['rmsd']:.2f} TM={r['tm_score']:.3f}{marker}")

    # Link best model to registry
    best = best_runs[0]
    print(f"\nPromoting '{best['name']}' to model registry...")

    # Find the artifact from this run
    run = api.run(f"{ENTITY}/{PROJECT}/{best['run_id']}")
    model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]

    if not model_artifacts:
        print("No model artifact found for best run. Check training script.")
        return

    artifact = model_artifacts[-1]  # latest version
    print(f"  Artifact: {artifact.name} (v{artifact.version})")

    # Link to registry with aliases
    artifact.link(f"{ENTITY}/wandb-registry-model/{REGISTRY_NAME}")
    print(f"  Linked to registry: {REGISTRY_NAME}")

    # Add aliases
    artifact.aliases.append("production")
    artifact.aliases.append("best")
    artifact.save()
    print(f"  Aliases: production, best")

    # If there's a second-best, mark as staging
    if len(best_runs) > 1:
        second = best_runs[1]
        run2 = api.run(f"{ENTITY}/{PROJECT}/{second['run_id']}")
        artifacts2 = [a for a in run2.logged_artifacts() if a.type == "model"]
        if artifacts2:
            art2 = artifacts2[-1]
            art2.link(f"{ENTITY}/wandb-registry-model/{REGISTRY_NAME}")
            art2.aliases.append("staging")
            art2.save()
            print(f"  Staging: '{second['name']}' (RMSD={second['rmsd']:.2f})")

    print(f"\nRegistry updated!")
    print(f"View: https://wandb.ai/{ENTITY}/registry/model/{REGISTRY_NAME}")


if __name__ == "__main__":
    main()
