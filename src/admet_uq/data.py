"""TDC ADMET data: task registry, download, curation, and target transforms.

Curation runs at download time and writes a clean parquet per task:
``normalise columns -> N-oxide SMILES corrections -> parse ->
ChEMBL standardize + get_parent -> parent InChIKey``.

Repeated measurements of one compound are kept as separate rows. The
``compound_id`` column, the parent InChIKey, groups them.

``load_task`` returns the target in its original units.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Canonical 22-task TDC ADMET matrix: name -> "regression" | "binary".
TASKS: dict[str, str] = {
    # Absorption
    "caco2_wang": "regression",
    "hia_hou": "binary",
    "pgp_broccatelli": "binary",
    "bioavailability_ma": "binary",
    "lipophilicity_astrazeneca": "regression",
    "solubility_aqsoldb": "regression",
    # Distribution
    "bbb_martins": "binary",
    "ppbr_az": "regression",
    "vdss_lombardo": "regression",
    # Metabolism
    "cyp2c9_veith": "binary",
    "cyp2d6_veith": "binary",
    "cyp3a4_veith": "binary",
    "cyp2c9_substrate_carbonmangels": "binary",
    "cyp2d6_substrate_carbonmangels": "binary",
    "cyp3a4_substrate_carbonmangels": "binary",
    # Excretion
    "half_life_obach": "regression",
    "clearance_microsome_az": "regression",
    "clearance_hepatocyte_az": "regression",
    # Toxicity
    "herg": "binary",
    "ames": "binary",
    "dili": "binary",
    "ld50_zhu": "regression",
}

# Per-task TDC leaderboard metric: name -> (metric, direction). HPO selection +
# primary reporting criterion. ``ap`` = average precision (AUPRC).
PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "caco2_wang": ("mae", "minimize"),
    "hia_hou": ("auroc", "maximize"),
    "pgp_broccatelli": ("auroc", "maximize"),
    "bioavailability_ma": ("auroc", "maximize"),
    "lipophilicity_astrazeneca": ("mae", "minimize"),
    "solubility_aqsoldb": ("mae", "minimize"),
    "bbb_martins": ("auroc", "maximize"),
    "ppbr_az": ("mae", "minimize"),
    "vdss_lombardo": ("spearman", "maximize"),
    "cyp2c9_veith": ("ap", "maximize"),
    "cyp2d6_veith": ("ap", "maximize"),
    "cyp3a4_veith": ("ap", "maximize"),
    "cyp2c9_substrate_carbonmangels": ("ap", "maximize"),
    "cyp2d6_substrate_carbonmangels": ("ap", "maximize"),
    "cyp3a4_substrate_carbonmangels": ("auroc", "maximize"),
    "half_life_obach": ("spearman", "maximize"),
    "clearance_microsome_az": ("spearman", "maximize"),
    "clearance_hepatocyte_az": ("spearman", "maximize"),
    "herg": ("auroc", "maximize"),
    "ames": ("auroc", "maximize"),
    "dili": ("auroc", "maximize"),
    "ld50_zhu": ("mae", "minimize"),
}

# Regression targets trained in log space and reported in original units.
#
# PK variability is multiplicative (enzyme activity, flow, permeation), so
# clearance and volume of distribution are log-normal, and half-life, being the
# ratio t_1/2 = ln2 * Vd/CL, follows.
LOG_TARGET_TASKS: frozenset[str] = frozenset(
    {
        "vdss_lombardo",
        "half_life_obach",
        "clearance_microsome_az",
        "clearance_hepatocyte_az",
    }
)

# Harvard Dataverse file id per task.
_DATAVERSE_FILE_ID: dict[str, int] = {
    "caco2_wang": 4259569,
    "hia_hou": 4259591,
    "pgp_broccatelli": 4259597,
    "bioavailability_ma": 4259567,
    "lipophilicity_astrazeneca": 4259595,
    "solubility_aqsoldb": 4259610,
    "bbb_martins": 4259566,
    "ppbr_az": 6413140,
    "vdss_lombardo": 4267387,
    "cyp2c9_veith": 4259577,
    "cyp2d6_veith": 4259580,
    "cyp3a4_veith": 4259582,
    "cyp2c9_substrate_carbonmangels": 4259584,
    "cyp2d6_substrate_carbonmangels": 4259578,
    "cyp3a4_substrate_carbonmangels": 4259581,
    "half_life_obach": 4266799,
    "clearance_microsome_az": 4266186,
    "clearance_hepatocyte_az": 4266187,
    "herg": 4259588,
    "ames": 4259564,
    "dili": 4259585,
    "ld50_zhu": 4267146,
}

_DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

# AqSolDB ships two rows whose SMILES are non-physical encodings of pyridine
# N-oxides ([NH+2]/[CH-]) that RDKit rejects. The source Drug_IDs name the real
# compounds, so the correct canonical structures are substituted (formulas match
# the names), keeping the solubility measurements. task -> {invalid: corrected}.
_SMILES_CORRECTIONS: dict[str, dict[str, str]] = {
    "solubility_aqsoldb": {
        # Drug_ID "3-methyl-n-oxidepyridine" -> 3-methylpyridine N-oxide (C6H7NO)
        "CC1=CC=C[NH+2]([O-])[CH-]1": "Cc1ccc[n+]([O-])c1",
        # Drug_ID "n-oxidenicotinic acid"  -> nicotinic acid N-oxide (C6H5NO3)
        "O=C(O)C1=C[NH+2]([O-])[CH-]C=C1": "O=C(O)c1ccc[n+]([O-])c1",
    }
}


def primary_metric(task: str) -> tuple[str, str]:
    """Return the leaderboard ``(metric, direction)`` pair for ``task``."""
    if task not in PRIMARY_METRIC:
        raise KeyError(f"unknown TDC ADMET task: {task!r}")
    return PRIMARY_METRIC[task]


def target_transform(task: str) -> str:
    """Return the training-target transform for ``task``: ``"log"`` or ``"identity"``."""
    if task not in TASKS:
        raise KeyError(f"unknown TDC ADMET task: {task!r}")
    return "log" if task in LOG_TARGET_TASKS else "identity"


def transform_target(task: str, y: np.ndarray) -> np.ndarray:
    """Map a raw target into the space the model is trained in."""
    y = np.asarray(y, dtype=np.float64)
    if target_transform(task) == "identity":
        return y
    if y.size and y.min() <= 0:
        raise ValueError(f"{task}: log transform needs positive targets, got {y.min()}")
    return np.log(y)


def inverse_target(task: str, y: np.ndarray) -> np.ndarray:
    """Map a model output back to the task's original units, for reporting.

    ``exp`` of a predicted log-mean is the geometric mean, so this returns the
    median of the implied log-normal, not its mean.
    """
    if target_transform(task) == "identity":
        return np.asarray(y, dtype=np.float64)
    return np.exp(np.asarray(y, dtype=np.float64))


def _download_tsv(task: str, raw_dir: Path) -> Path:
    """Fetch the raw TSV from Harvard Dataverse, caching under ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{task}.tab"
    if out.exists():
        return out
    url = _DATAVERSE_URL.format(file_id=_DATAVERSE_FILE_ID[task])
    log.info("[%s] downloading %s", task, url)
    req = urllib.request.Request(url, headers={"User-Agent": "admet-uq/1.0.0"})
    with urllib.request.urlopen(req) as resp, out.open("wb") as f:
        f.write(resp.read())
    return out


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TDC's heterogeneous column names to ``(smiles, y)``."""
    smiles_col = next(
        (col for col in ("Drug", "X", "smiles", "SMILES", "Smiles") if col in df.columns), None
    )
    label_col = next((col for col in ("Y", "y", "label", "activity") if col in df.columns), None)
    if smiles_col is None or label_col is None:
        raise ValueError(f"unexpected TDC columns: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "smiles": df[smiles_col].astype(str),
            "y": pd.to_numeric(df[label_col], errors="coerce").astype("float64"),
        }
    )
    return out.dropna(subset=["smiles", "y"]).reset_index(drop=True)


def _apply_smiles_corrections(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Replace the known-invalid source SMILES with curated valid structures."""
    fixes = _SMILES_CORRECTIONS.get(task)
    if not fixes:
        return df
    df = df.copy()
    n = int(df["smiles"].isin(fixes).sum())
    df["smiles"] = df["smiles"].replace(fixes)
    if n:
        log.info("[%s] applied %d SMILES corrections", task, n)
    return df


def _curate(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, int]:
    """Standardize and de-salt each row, adding ``compound_id``.

    ``standardize_mol`` neutralizes, normalizes functional groups and kekulizes;
    ``get_parent_mol`` strips salts and solvents from a curated list while
    leaving genuine mixtures intact. The parent's Standard InChIKey becomes
    ``compound_id``.
    """
    from chembl_structure_pipeline import standardizer
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    raw_smiles, smiles, compound_id, ys = [], [], [], []
    n_fail = 0
    for smi, y in zip(df["smiles"], df["y"], strict=True):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            n_fail += 1
            continue
        try:
            parent, _ = standardizer.get_parent_mol(standardizer.standardize_mol(mol))
            psmi = Chem.MolToSmiles(parent)
            # A few pathological structures (odd aromatic perception, some
            # organometallics) yield a parent SMILES RDKit cannot re-parse; drop
            # them so the parquet only ever holds usable, round-trippable SMILES.
            if Chem.MolFromSmiles(psmi) is None:
                n_fail += 1
                continue
            cid = Chem.InchiToInchiKey(Chem.MolToInchi(parent))
        except Exception:  # noqa: BLE001 - a curation failure drops the row (counted)
            n_fail += 1
            continue
        raw_smiles.append(smi)
        smiles.append(psmi)
        compound_id.append(cid)
        ys.append(float(y))
    out = pd.DataFrame(
        {"raw_smiles": raw_smiles, "smiles": smiles, "compound_id": compound_id, "y": ys}
    )
    return out, n_fail


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically (tmp + rename) for concurrent reads."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def download_task(task: str, data_root: str | Path) -> Path:
    """Download and curate one TDC task; return the path to its ``data.parquet``.

    Writes ``<data_root>/<task>/data.parquet`` with columns
    ``raw_smiles, smiles (curated parent), compound_id (parent InChIKey), y``,
    plus a ``meta.json`` recording the row and compound counts.
    """
    if task not in TASKS:
        raise KeyError(f"unknown TDC ADMET task: {task!r}")
    task_dir = Path(data_root) / task
    out = task_dir / "data.parquet"

    tsv = _download_tsv(task, task_dir / "raw")
    norm = _apply_smiles_corrections(_normalise(pd.read_csv(tsv, sep="\t")), task)
    n_raw = len(norm)
    curated, n_fail = _curate(norm, task)
    if n_raw and n_fail / n_raw > 0.01:
        raise RuntimeError(f"[{task}] {n_fail}/{n_raw} rows failed curation, above the 1% limit")

    task_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(curated, out)
    (task_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": task,
                "task_type": TASKS[task],
                "file_id": _DATAVERSE_FILE_ID[task],
                "n_raw": n_raw,
                "n_curation_fail": n_fail,
                "n_rows": len(curated),
                "n_compounds": int(curated["compound_id"].nunique()),
                "n_multicomponent": int(curated["smiles"].str.contains(".", regex=False).sum()),
            },
            indent=2,
        )
    )
    log.info(
        "[%s] %d rows / %d compounds -> %s (task_type=%s)",
        task,
        len(curated),
        curated["compound_id"].nunique(),
        out,
        TASKS[task],
    )
    return out


def load_task(task: str, data_root: str | Path) -> pd.DataFrame:
    """Load a curated TDC task: columns ``smiles``, ``compound_id``, ``y``."""
    if task not in TASKS:
        raise KeyError(f"unknown TDC ADMET task: {task!r}")
    parquet = Path(data_root) / task / "data.parquet"
    if not parquet.is_file():
        download_task(task, data_root)
    return pd.read_parquet(parquet)[["smiles", "compound_id", "y"]].reset_index(drop=True)


def _cli(task: str, data_root: Path = Path("data/tdc_admet")) -> None:
    """Download + curate a TDC ADMET task (use ``all`` for every task)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in list(TASKS) if task == "all" else [task]:
        download_task(name, data_root)


if __name__ == "__main__":
    import tyro

    tyro.cli(_cli)
