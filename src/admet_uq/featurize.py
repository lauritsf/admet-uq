"""Morgan-count fingerprint concatenated with descriptastorus RDKit2DNormalized.

The descriptor block is CDF-normalized to ~[0,1] against a fixed external
reference distribution.

Features are computed on the curated parent SMILES.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from admet_uq import data as data_mod

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_DIM = 2048
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_DIM)

# descriptastorus ``.process(smiles)`` returns ``[ok_flag, *descriptors]``.
_RDKIT2D = rdNormalizedDescriptors.RDKit2DNormalized()
N_RDKIT = len(_RDKIT2D.process("C")) - 1
FEATURE_DIM = MORGAN_DIM + N_RDKIT
# Cache-directory tag: the fingerprint and descriptor dimensions.
FEATURE_SPEC = f"morgan_count_{MORGAN_DIM}_r{MORGAN_RADIUS}__rdkit2dnorm_{N_RDKIT}"


def featurize_smiles(smiles: str) -> np.ndarray | None:
    """Return the ``(FEATURE_DIM,)`` float32 vector, or ``None`` on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    counts = _MORGAN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    row = _RDKIT2D.process(smiles)
    if not row[0]:
        return None
    # Gasteiger partial charges are undefined for metals; 0.5 is the CDF median.
    desc = np.nan_to_num(np.asarray(row[1:], dtype=np.float32), nan=0.5)
    return np.concatenate([counts, desc])


def featurize_task(
    task: str,
    data_root: str | Path,
    cache_root: str | Path,
) -> dict[str, np.ndarray]:
    """Featurize a task, caching to ``<cache_root>/<spec>/<task>.npz``.

    Returns a dict with ``X`` ``(N, FEATURE_DIM)`` float32, ``y`` ``(N,)``
    float32, ``compound_id`` ``(N,)`` str, ``smiles`` ``(N,)`` str. Rows that
    fail featurization are dropped (all four arrays stay aligned).
    """
    cache_dir = Path(cache_root) / FEATURE_SPEC
    cache_path = cache_dir / f"{task}.npz"
    if cache_path.is_file():
        npz = np.load(cache_path, allow_pickle=False)
        return {key: npz[key] for key in ("X", "y", "compound_id", "smiles")}

    df = data_mod.load_task(task, data_root)
    feats, labels, compound_ids, smiles_kept = [], [], [], []
    for smi, cid, y in zip(df["smiles"], df["compound_id"], df["y"], strict=True):
        vec = featurize_smiles(smi)
        if vec is None:
            continue
        feats.append(vec)
        labels.append(y)
        compound_ids.append(cid)
        smiles_kept.append(smi)
    n_drop = len(df) - len(feats)
    if n_drop:
        log.info("[%s] %d/%d rows failed featurization (dropped)", task, n_drop, len(df))

    out = {
        "X": np.asarray(feats, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.float32),
        "compound_id": np.asarray(compound_ids),
        "smiles": np.asarray(smiles_kept),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **out)
    log.info("[%s] cached %d x %d features -> %s", task, len(feats), FEATURE_DIM, cache_path)
    return out


def _cli(
    task: str,
    data_root: Path = Path("data/tdc_admet"),
    cache_root: Path = Path("data/features"),
) -> None:
    """Featurize + cache a TDC ADMET task (use ``all`` for every task)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("feature spec: %s (dim=%d)", FEATURE_SPEC, FEATURE_DIM)
    for name in list(data_mod.TASKS) if task == "all" else [task]:
        featurize_task(name, data_root, cache_root)


if __name__ == "__main__":
    import tyro

    tyro.cli(_cli)
