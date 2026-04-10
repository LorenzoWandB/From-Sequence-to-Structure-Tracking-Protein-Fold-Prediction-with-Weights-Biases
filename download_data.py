#!/usr/bin/env python3
"""
ProteusAI — Download Real Protein Data for ESMFold
====================================================
Downloads amino acid sequences from UniProt and reference structures
from AlphaFold DB for 30 drug targets. Splits into train/val sets.

ESMFold handles sequence → structure end-to-end, so we only need
sequences as input and AlphaFold structures as ground truth.

Usage:
    uv run python download_data.py
"""

import json
import os
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path("data")
STRUCT_DIR = DATA_DIR / "structures"

# 30 real drug targets across therapeutic areas
PROTEINS = {
    # ── Oncology (15) ──
    "TP53": {"uniprot": "P04637", "area": "Oncology", "role": "Tumor suppressor"},
    "AKT1": {"uniprot": "P31749", "area": "Oncology", "role": "Kinase / PI3K pathway"},
    "PIK3CA": {"uniprot": "P42336", "area": "Oncology", "role": "Lipid kinase"},
    "EGFR": {"uniprot": "P00533", "area": "Oncology", "role": "Receptor tyrosine kinase"},
    "BRAF": {"uniprot": "P15056", "area": "Oncology", "role": "Ser/Thr kinase"},
    "CDK2": {"uniprot": "P24941", "area": "Oncology", "role": "Cell cycle kinase"},
    "KRAS": {"uniprot": "P01116", "area": "Oncology", "role": "GTPase"},
    "ABL1": {"uniprot": "P00519", "area": "Oncology", "role": "Tyrosine kinase"},
    "SRC": {"uniprot": "P12931", "area": "Oncology", "role": "Proto-oncogene kinase"},
    "HER2": {"uniprot": "P04626", "area": "Oncology", "role": "Receptor tyrosine kinase"},
    "MDM2": {"uniprot": "Q00987", "area": "Oncology", "role": "p53 regulator"},
    "BCL2": {"uniprot": "P10415", "area": "Oncology", "role": "Apoptosis regulator"},
    "PTEN": {"uniprot": "P60484", "area": "Oncology", "role": "Phosphatase / tumor suppressor"},
    "CDK4": {"uniprot": "P11802", "area": "Oncology", "role": "Cell cycle kinase"},
    "AURORA_A": {"uniprot": "O14965", "area": "Oncology", "role": "Mitotic kinase"},

    # ── Rare Disease (5) ──
    "GBA1": {"uniprot": "P04062", "area": "Rare Disease", "role": "Glucocerebrosidase (Gaucher)"},
    "CFTR": {"uniprot": "P13569", "area": "Rare Disease", "role": "Chloride channel (CF)"},
    "NF1": {"uniprot": "P21359", "area": "Rare Disease", "role": "RasGAP (Neurofibromatosis)"},
    "GLA": {"uniprot": "P06280", "area": "Rare Disease", "role": "Alpha-galactosidase (Fabry)"},
    "SMN1": {"uniprot": "Q16637", "area": "Rare Disease", "role": "RNA processing (SMA)"},

    # ── Infectious Disease (5) ──
    "SPIKE": {"uniprot": "P0DTC2", "area": "Infectious Disease", "role": "SARS-CoV-2 Spike"},
    "HIV_RT": {"uniprot": "P03366", "area": "Infectious Disease", "role": "HIV-1 reverse transcriptase"},
    "NS5B": {"uniprot": "Q99IB2", "area": "Infectious Disease", "role": "HCV RNA polymerase"},
    "DHFR_PF": {"uniprot": "P13922", "area": "Infectious Disease", "role": "P.falciparum DHFR (malaria)"},
    "NEURAM": {"uniprot": "P03472", "area": "Infectious Disease", "role": "Influenza neuraminidase"},

    # ── Metabolic / Other (5) ──
    "JAK2": {"uniprot": "O60674", "area": "Metabolic", "role": "Cytokine signaling kinase"},
    "ERK2": {"uniprot": "P28482", "area": "Metabolic", "role": "MAP kinase"},
    "RAF1": {"uniprot": "P04049", "area": "Metabolic", "role": "Ser/Thr kinase"},
    "FGFR2": {"uniprot": "P21802", "area": "Metabolic", "role": "Growth factor receptor"},
    "ALK": {"uniprot": "Q9UM73", "area": "Metabolic", "role": "Receptor tyrosine kinase"},
}


def fetch_sequence(uniprot_id: str) -> str | None:
    """Fetch amino acid sequence from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            return "".join(lines[1:])
    except Exception as e:
        print(f"    UniProt error: {e}")
    return None


def fetch_alphafold_structure(gene: str, uniprot_id: str) -> dict | None:
    """Fetch reference structure from AlphaFold DB (ground truth for evaluation)."""
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        resp = requests.get(api_url, timeout=30)
        if resp.status_code != 200:
            return None

        data = resp.json()[0]
        pdb_url = data.get("pdbUrl")
        if not pdb_url:
            return None

        pdb_resp = requests.get(pdb_url, timeout=60)
        if pdb_resp.status_code != 200:
            return None

        # Save PDB
        pdb_path = STRUCT_DIR / f"{gene}_{uniprot_id}.pdb"
        pdb_path.write_text(pdb_resp.text)

        # Extract Cα coordinates + pLDDT from B-factor column
        coords, plddts = [], []
        for line in pdb_resp.text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                plddts.append(float(line[60:66]))

        if not coords:
            return None

        np.save(STRUCT_DIR / f"{gene}_coords.npy", np.array(coords, dtype=np.float32))
        np.save(STRUCT_DIR / f"{gene}_plddt.npy", np.array(plddts, dtype=np.float32))

        return {
            "pdb_path": str(pdb_path),
            "seq_len_struct": len(coords),
            "mean_plddt": float(np.mean(plddts)),
        }
    except Exception as e:
        print(f"    AlphaFold error: {e}")
        return None


def main():
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    print(f"Downloading {len(PROTEINS)} drug target proteins...\n")

    for gene, info in PROTEINS.items():
        uid = info["uniprot"]
        print(f"  [{gene}] {info['role']} ({uid})...", end=" ")

        # Sequence from UniProt
        seq = fetch_sequence(uid)
        if not seq:
            print("SKIP (no sequence)")
            continue

        # Reference structure from AlphaFold DB
        struct = fetch_alphafold_structure(gene, uid)

        entry = {
            "gene": gene,
            "uniprot_id": uid,
            "area": info["area"],
            "role": info["role"],
            "sequence": seq,
            "seq_len": len(seq),
        }

        if struct:
            entry.update(struct)
            print(f"OK — {len(seq)} aa, struct {struct['seq_len_struct']} res, "
                  f"pLDDT={struct['mean_plddt']:.1f}")
        else:
            print(f"OK — {len(seq)} aa (no AlphaFold structure)")

        manifest.append(entry)

    # Split into train (70%) and val (30%) — held-out proteins for honest eval
    np.random.seed(42)
    indices = np.random.permutation(len(manifest))
    split = int(0.7 * len(manifest))
    for i, idx in enumerate(indices):
        manifest[idx]["split"] = "train" if i < split else "val"

    train_genes = [m["gene"] for m in manifest if m["split"] == "train"]
    val_genes = [m["gene"] for m in manifest if m["split"] == "val"]

    # Save manifest
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Downloaded {len(manifest)}/{len(PROTEINS)} proteins")
    print(f"Train ({len(train_genes)}): {', '.join(train_genes)}")
    print(f"Val   ({len(val_genes)}):  {', '.join(val_genes)}")
    print(f"Manifest: {manifest_path}")
    print(f"Structures: {STRUCT_DIR}")


if __name__ == "__main__":
    main()
