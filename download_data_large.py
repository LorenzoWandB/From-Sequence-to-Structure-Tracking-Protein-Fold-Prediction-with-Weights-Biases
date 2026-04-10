#!/usr/bin/env python3
"""
ProteusAI — Download Large Protein Dataset for Training
=========================================================
Downloads ~500 diverse proteins from the PDB/AlphaFold DB for real training
with visible learning curves. Also keeps the 30 drug targets as val set.

Usage:
    uv run python download_data_large.py
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path("data_large")
STRUCT_DIR = DATA_DIR / "structures"

# ── Original 30 drug targets (used as VAL set) ──
DRUG_TARGETS = {
    "TP53": "P04637", "AKT1": "P31749", "PIK3CA": "P42336", "EGFR": "P00533",
    "BRAF": "P15056", "CDK2": "P24941", "KRAS": "P01116", "ABL1": "P00519",
    "SRC": "P12931", "HER2": "P04626", "MDM2": "Q00987", "BCL2": "P10415",
    "PTEN": "P60484", "CDK4": "P11802", "AURORA_A": "O14965",
    "GBA1": "P04062", "CFTR": "P13569", "NF1": "P21359", "GLA": "P06280", "SMN1": "Q16637",
    "SPIKE": "P0DTC2", "HIV_RT": "P03366", "NS5B": "Q99IB2", "DHFR_PF": "P13922", "NEURAM": "P03472",
    "JAK2": "O60674", "ERK2": "P28482", "RAF1": "P04049", "FGFR2": "P21802", "ALK": "Q9UM73",
}

# ── 500 diverse well-characterized human proteins for TRAIN set ──
# These are well-studied human proteins from Swiss-Prot with high-quality AlphaFold predictions
TRAIN_UNIPROT_IDS = [
    # Kinases (50)
    "P00519", "P06239", "P07332", "P07333", "P08631", "P08922", "P09619", "P10721",
    "P11362", "P12931", "P16234", "P17252", "P17612", "P19174", "P22607", "P23458",
    "P24941", "P28482", "P29317", "P29320", "P30291", "P30530", "P31749", "P31751",
    "P36507", "P36888", "P42336", "P42345", "P43405", "P45983", "P45984", "P49137",
    "P49841", "P51812", "P52564", "P53350", "P53779", "P54646", "P56192", "Q00534",
    "Q00535", "Q02156", "Q02750", "Q05397", "Q06187", "Q08881", "Q13131", "Q13153",
    "Q13177", "Q13188",
    # Phosphatases (30)
    "P16885", "P18031", "P23467", "P29350", "P36873", "P41743", "P49023", "P60484",
    "Q06124", "Q13289", "Q14289", "Q16539", "Q16584", "Q16611", "Q16672", "Q92835",
    "Q99590", "Q9H3S7", "Q9NRD5", "Q9NYF8", "Q9UBR1", "Q9UDY2", "Q9UGJ0", "Q9UHD2",
    "Q9UKI8", "Q9ULW4", "Q9Y243", "Q9Y4K3", "Q9Y5S2", "Q9Y6R4",
    # Proteases (30)
    "P00734", "P00740", "P00742", "P00748", "P03952", "P03956", "P07339", "P07858",
    "P08236", "P08253", "P08254", "P09668", "P09871", "P10144", "P14091", "P15085",
    "P15169", "P20742", "P22894", "P24821", "P25774", "P39060", "P43234", "P56817",
    "Q07820", "Q13510", "Q14703", "Q99538", "Q9UBR2", "Q9Y5S5",
    # Transcription factors (40)
    "P01100", "P01106", "P01137", "P03372", "P04150", "P04637", "P05412", "P05771",
    "P06400", "P08047", "P09874", "P10242", "P10275", "P10276", "P10826", "P11473",
    "P13569", "P14921", "P15407", "P17275", "P17947", "P19838", "P20226", "P20823",
    "P21359", "P23297", "P25490", "P26358", "P28749", "P29353", "P36956", "P38398",
    "P38936", "P42224", "P42226", "P42229", "P46937", "P49711", "P51531", "P55317",
    # Receptors (50)
    "P00533", "P01588", "P02545", "P04626", "P04629", "P04637", "P06213", "P07355",
    "P07948", "P08069", "P08581", "P08684", "P09038", "P10145", "P10415", "P11166",
    "P11511", "P12821", "P13569", "P14210", "P15056", "P16220", "P16234", "P17181",
    "P18627", "P19793", "P20338", "P20848", "P21397", "P21554", "P21802", "P22888",
    "P23759", "P24385", "P24941", "P25445", "P27169", "P29474", "P30556", "P32004",
    "P35228", "P35367", "P35372", "P35968", "P36888", "P40189", "P40763", "P42262",
    "P43489", "P48061",
    # Enzymes / metabolic (50)
    "P00325", "P00352", "P00374", "P00390", "P00439", "P00441", "P00480", "P00505",
    "P00558", "P00568", "P00918", "P01009", "P01011", "P01024", "P01308", "P02647",
    "P02649", "P02671", "P02675", "P02679", "P02741", "P02751", "P02768", "P02787",
    "P03886", "P04040", "P04062", "P04075", "P04083", "P04156", "P04217", "P04406",
    "P04899", "P05091", "P05155", "P05164", "P05177", "P05186", "P06280", "P06307",
    "P06727", "P07195", "P07237", "P07288", "P07305", "P07339", "P07384", "P07550",
    "P07602", "P07686",
    # Structural / cytoskeletal (30)
    "P02452", "P02461", "P02462", "P02545", "P04264", "P05023", "P05556", "P06396",
    "P06753", "P07437", "P07900", "P07942", "P08107", "P08133", "P08238", "P08670",
    "P09211", "P09382", "P09493", "P09960", "P10645", "P10909", "P11142", "P11413",
    "P11498", "P12081", "P12277", "P12814", "P13010", "P13489",
    # Channels / transporters (30)
    "P02549", "P05023", "P05067", "P05154", "P05362", "P06756", "P07996", "P08195",
    "P08217", "P08247", "P08684", "P10909", "P11166", "P11279", "P11717", "P12277",
    "P13569", "P13637", "P14210", "P14416", "P14867", "P17342", "P18507", "P19429",
    "P21453", "P21917", "P22303", "P22459", "P23415", "P23634",
    # Immune system (40)
    "P01374", "P01375", "P01579", "P01583", "P01584", "P01588", "P01589", "P02743",
    "P02747", "P04141", "P04637", "P05112", "P05113", "P05114", "P05121", "P05362",
    "P05771", "P06213", "P06239", "P07766", "P09564", "P09601", "P09960", "P10145",
    "P10147", "P10415", "P10586", "P10589", "P10643", "P10721", "P11362", "P11473",
    "P13232", "P13500", "P14174", "P14780", "P15169", "P15692", "P16220", "P16284",
]


def fetch_sequence(uniprot_id):
    try:
        resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            return "".join(lines[1:])
    except:
        pass
    return None


def fetch_structure(gene_or_id, uniprot_id):
    try:
        resp = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}", timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()[0]
        pdb_url = data.get("pdbUrl")
        if not pdb_url:
            return None
        pdb_resp = requests.get(pdb_url, timeout=30)
        if pdb_resp.status_code != 200:
            return None

        pdb_path = STRUCT_DIR / f"{gene_or_id}_{uniprot_id}.pdb"
        pdb_path.write_text(pdb_resp.text)

        coords, plddts = [], []
        for line in pdb_resp.text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                plddts.append(float(line[60:66]))

        if not coords:
            return None

        np.save(STRUCT_DIR / f"{gene_or_id}_coords.npy", np.array(coords, dtype=np.float32))
        np.save(STRUCT_DIR / f"{gene_or_id}_plddt.npy", np.array(plddts, dtype=np.float32))

        return {
            "pdb_path": str(pdb_path),
            "seq_len_struct": len(coords),
            "mean_plddt": float(np.mean(plddts)),
        }
    except:
        return None


def main():
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    # ── Download drug targets (val set) ──
    print(f"=== Downloading {len(DRUG_TARGETS)} drug targets (val set) ===")
    for gene, uid in DRUG_TARGETS.items():
        seq = fetch_sequence(uid)
        if not seq:
            continue
        struct = fetch_structure(gene, uid)
        entry = {"gene": gene, "uniprot_id": uid, "sequence": seq, "seq_len": len(seq),
                 "split": "val", "area": "drug-target"}
        if struct:
            entry.update(struct)
            print(f"  [val] {gene}: {len(seq)} aa, pLDDT={struct['mean_plddt']:.1f}")
        else:
            print(f"  [val] {gene}: {len(seq)} aa (no structure)")
        manifest.append(entry)

    # ── Download diverse proteins (train set) ──
    # Deduplicate and remove any that overlap with drug targets
    train_ids = list(set(TRAIN_UNIPROT_IDS) - set(DRUG_TARGETS.values()))
    random.seed(42)
    random.shuffle(train_ids)

    print(f"\n=== Downloading {len(train_ids)} diverse proteins (train set) ===")
    downloaded = 0
    failed = 0
    for i, uid in enumerate(train_ids):
        if downloaded >= 500:
            break
        seq = fetch_sequence(uid)
        if not seq or len(seq) > 300 or len(seq) < 50:
            # Skip very long (OOM) and very short proteins
            failed += 1
            continue

        struct = fetch_structure(uid, uid)
        if not struct:
            failed += 1
            continue

        entry = {"gene": uid, "uniprot_id": uid, "sequence": seq, "seq_len": len(seq),
                 "split": "train", "area": "diverse"}
        entry.update(struct)
        manifest.append(entry)
        downloaded += 1

        if downloaded % 25 == 0:
            print(f"  [train] {downloaded} downloaded, {failed} skipped...")

    train_count = sum(1 for m in manifest if m["split"] == "train")
    val_count = sum(1 for m in manifest if m["split"] == "val")

    # Save manifest
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Train: {train_count} proteins")
    print(f"Val: {val_count} proteins (drug targets)")
    print(f"Total: {len(manifest)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
