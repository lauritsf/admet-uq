"""Butina-cluster nested cross-validation splits.

Folds are assigned at the compound level: all rows sharing a ``compound_id``
land in the same fold. The id is the parent InChIKey, which replicates,
tautomers and salt forms of one compound all share.

The grouping unit is a Taylor-Butina cluster over binary Morgan fingerprints
(r=2, 2048 bits) at Tanimoto distance cutoff 0.5.

Nested CV = an outer k-fold (each fold a held-out test set) with, inside each
``outer_train``, an inner k-fold used for HPO config selection. Clusters are
recomputed inside each ``outer_train``. All indices returned are into the input
arrays, which must be row-aligned with the feature matrix.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
# Tanimoto distance cutoff. Points within this distance of a centroid join its
# cluster.
BUTINA_CUTOFF = 0.5
# Fraction of each ``outer_train``'s observations reserved for post-hoc calibration.
CALIBRATION_FRACTION = 0.2
# Sentinel values in the per-outer-fold ``inner`` column. Non-negative entries
# are inner-fold indices; these two mark compounds that are not in the inner CV
# at all.
TEST = -1
CALIBRATION = -2
# Rows of the similarity matrix computed per block, so the float32 temporaries
# are ``3 * _SIM_BLOCK * n`` rather than three dense n x n arrays.
_SIM_BLOCK = 1024

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


def morgan_bits(smiles: str) -> np.ndarray:
    """Binary Morgan fingerprint of the molecule's largest fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(MORGAN_BITS, dtype=bool)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda frag: frag.GetNumHeavyAtoms())
        # Fragments come out unsanitized (sanitizing them can fail on charged
        # pieces of a genuine mixture), which leaves RingInfo uninitialized.
        # Morgan's default atom invariants read "is in ring", so initialize it
        # explicitly rather than paying for a full sanitization.
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    return _MORGAN.GetFingerprintAsNumPy(mol).astype(bool)


def _compound_fingerprints(
    compound_ids: np.ndarray, smiles: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """One fingerprint row per unique compound, from its first-seen SMILES."""
    seen: dict[str, int] = {}
    rows: list[np.ndarray] = []
    for cid, smi in zip(compound_ids, smiles, strict=True):
        if cid in seen:
            continue
        seen[cid] = len(rows)
        rows.append(morgan_bits(str(smi)))
    return list(seen), np.asarray(rows, dtype=bool)


def _neighbour_matrix(fp_bits: np.ndarray, cutoff: float) -> np.ndarray:
    """Boolean ``n x n`` matrix of "within ``cutoff`` Tanimoto distance"."""
    n = len(fp_bits)
    counts = fp_bits.sum(axis=1).astype(np.float32)
    fp_float = fp_bits.astype(np.float32)
    threshold = 1.0 - cutoff
    neighbours = np.empty((n, n), dtype=bool)
    for start in range(0, n, _SIM_BLOCK):
        stop = min(start + _SIM_BLOCK, n)
        intersection = fp_float[start:stop] @ fp_float.T
        union = counts[start:stop, None] + counts[None, :] - intersection
        similarity = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        neighbours[start:stop] = similarity >= threshold
    # Tanimoto scores an all-zero fingerprint (unparseable SMILES) 0.0 against itself.
    np.fill_diagonal(neighbours, True)
    return neighbours


def butina_labels(fp_bits: np.ndarray, cutoff: float, seed: int) -> np.ndarray:
    """Taylor-Butina cluster label per row.

    Points are visited most-neighbours-first, and each unassigned one claims its
    unassigned neighbours as a cluster. Ties in that order are broken by a
    seeded shuffle rather than by row index.
    """
    neighbours = _neighbour_matrix(fp_bits, cutoff)
    rng = np.random.default_rng(seed)
    n_neighbours = neighbours.sum(axis=1)
    # lexsort applies the last key first.
    order = np.lexsort((rng.permutation(len(fp_bits)), -n_neighbours))

    assigned = np.zeros(len(fp_bits), dtype=bool)
    labels = np.full(len(fp_bits), -1, dtype=np.int32)
    next_label = 0
    for idx in order:
        if assigned[idx]:
            continue
        # The forced diagonal makes idx a member of its own cluster.
        members = np.flatnonzero(neighbours[idx] & ~assigned)
        assigned[members] = True
        labels[members] = next_label
        next_label += 1
    return labels


def _cluster_keys(cids: list[str], fp_bits: np.ndarray, cutoff: float, seed: int) -> dict[str, int]:
    labels = butina_labels(fp_bits, cutoff, seed)
    return dict(zip(cids, labels.tolist(), strict=True))


def _greedy_balanced_folds(groups: dict[int, list[str]], n_folds: int, seed: int) -> dict[str, int]:
    """Assign each cluster entirely to one fold, balancing compound counts.

    Largest groups first, each to the currently smallest fold. Ties are
    broken randomly, so fold index does not correlate with group size.
    """
    rng = random.Random(seed)
    group_list = list(groups.values())
    rng.shuffle(group_list)
    group_list.sort(key=len, reverse=True)  # stable: shuffled order kept within equal sizes
    sizes = [0] * n_folds
    fold_of: dict[str, int] = {}
    for group in group_list:
        order = list(range(n_folds))
        rng.shuffle(order)  # min() returns the first minimum -> random among ties
        fold = min(order, key=lambda idx: sizes[idx])
        for cid in group:
            fold_of[cid] = fold
        sizes[fold] += len(group)
    return fold_of


def _carve_calibration(
    cids: list[str],
    cluster: dict[str, int],
    weight: dict[str, int],
    frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split ``outer_train`` compounds into a calibration slice and a fitting pool.

    Whole Butina clusters move together, so split conformal keeps the
    exchangeability it assumes.

    Clusters are taken largest-first, so a large one is placed while the slice
    still has room. Ties among equal-sized clusters are shuffled from ``seed``.

    ``weight`` is each compound's replicate count, so the slice balances on
    observations, which is what the conformal quantile consumes.

    Raises:
        ValueError: if the realised slice falls below half the target.
    """
    groups: dict[int, list[str]] = defaultdict(list)
    for cid in cids:
        groups[cluster[cid]].append(cid)

    rng = random.Random(seed)
    group_list = list(groups.values())
    rng.shuffle(group_list)
    group_list.sort(key=lambda group: sum(weight[cid] for cid in group), reverse=True)

    target = round(frac * sum(weight[cid] for cid in cids))
    calibration: list[str] = []
    total = 0
    for group in group_list:
        size = sum(weight[cid] for cid in group)
        if total + size <= target:
            calibration.extend(group)
            total += size

    if total < target / 2:
        raise ValueError(
            f"calibration carve reached only {total} of {target} target observations: "
            f"one cluster dominates outer_train, so a {frac:.0%} cluster-level slice "
            f"is not achievable here."
        )
    chosen = set(calibration)
    return calibration, [cid for cid in cids if cid not in chosen]


def _fold_of(
    unique_cids: list[str], cluster: dict[str, int], n_folds: int, seed: int
) -> dict[str, int]:
    groups: dict[int, list[str]] = defaultdict(list)
    for cid in unique_cids:
        groups[cluster[cid]].append(cid)
    return _greedy_balanced_folds(groups, n_folds, seed)


@dataclass(frozen=True)
class OuterFold:
    """One outer CV fold, already split into its calibration and fitting parts.

    ``train_idx`` is the whole of ``outer_train``, partitioned into
    ``calibration_idx`` and ``fit_idx``.
    ``inner`` k-folds ``fit_idx`` rather than ``train_idx``.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    calibration_idx: np.ndarray
    fit_idx: np.ndarray
    inner: list[tuple[np.ndarray, np.ndarray]]  # (inner_train_idx, inner_val_idx) over fit_idx


def _compound_set_digest(compound_ids: Iterable[str]) -> str:
    joined = "\n".join(sorted({str(cid) for cid in compound_ids}))
    return hashlib.sha256(joined.encode()).hexdigest()


@dataclass(frozen=True)
class SplitAssignment:
    """A nested-CV split in its persisted, compound-level form.

    ``outer[i]`` is the outer fold of ``compound_ids[i]``. ``inner[i, k]`` is
    that compound's role within outer fold ``k``: ``TEST`` (once per row, since
    each compound is tested exactly once), ``CALIBRATION``, or the index of its
    inner fold in ``0 .. n_inner - 1``. ``cluster[i]`` is the global Butina
    cluster that produced the outer folds.
    """

    task: str
    compound_ids: list[str]
    cluster: np.ndarray  # (n_compounds,) global Butina cluster id
    outer: np.ndarray  # (n_compounds,) int
    inner: np.ndarray  # (n_compounds, n_outer) int
    n_outer: int
    n_inner: int
    seed: int
    cutoff: float
    calibration_fraction: float
    rdkit_version: str
    compound_digest: str

    def to_folds(self, compound_ids: np.ndarray) -> list[OuterFold]:
        """Materialize row indices into ``compound_ids`` (which may repeat).

        Raises:
            ValueError: if the compound set does not match the one the split
                was built on.
        """
        compound_ids = np.asarray(compound_ids)
        digest = _compound_set_digest(compound_ids.tolist())
        if digest != self.compound_digest:
            raise ValueError(
                f"[{self.task}] compound set does not match the stored split "
                f"({len(set(compound_ids.tolist()))} compounds, digest {digest[:12]} != "
                f"{self.compound_digest[:12]}). The split file was built on different data; "
                f"regenerate it rather than reusing it."
            )
        outer_of = dict(zip(self.compound_ids, self.outer.tolist(), strict=True))
        row_outer = np.array([outer_of[cid] for cid in compound_ids])

        folds: list[OuterFold] = []
        for outer in range(self.n_outer):
            test_idx = np.flatnonzero(row_outer == outer)
            train_idx = np.flatnonzero(row_outer != outer)
            inner_of = dict(zip(self.compound_ids, self.inner[:, outer].tolist(), strict=True))
            row_inner = np.array([inner_of[cid] for cid in compound_ids[train_idx]])

            calibration_idx = train_idx[np.flatnonzero(row_inner == CALIBRATION)]
            fit_idx = train_idx[np.flatnonzero(row_inner >= 0)]
            fit_inner = row_inner[row_inner >= 0]
            inner = [
                (
                    fit_idx[np.flatnonzero(fit_inner != inner_i)],
                    fit_idx[np.flatnonzero(fit_inner == inner_i)],
                )
                for inner_i in range(self.n_inner)
            ]
            folds.append(
                OuterFold(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    calibration_idx=calibration_idx,
                    fit_idx=fit_idx,
                    inner=inner,
                )
            )
        return folds


def assign_nested_cv(
    compound_ids: np.ndarray,
    smiles: np.ndarray,
    *,
    task: str = "",
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 0,
    cutoff: float = BUTINA_CUTOFF,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> SplitAssignment:
    """Compute the Butina-cluster, compound-grouped nested CV assignment.

    Inside each ``outer_train`` the order is carve, then k-fold: the dedicated
    slice comes out first and the inner CV partitions what is left, so the
    calibration slice is cluster-disjoint from everything the model is fitted on
    or selected with.

    Clustering runs once globally for the outer folds, then again inside each
    ``outer_train``. Butina is not stable under subsetting, so the two are not
    the same partition. Refitting inside ``outer_train`` keeps the held-out
    compounds out of the inner grouping, where a test compound acting as a
    centroid could otherwise pull two training compounds into one cluster.
    """
    compound_ids = np.asarray(compound_ids)
    smiles = np.asarray(smiles)
    unique_cids, fp_bits = _compound_fingerprints(compound_ids, smiles)
    row_of_cid = {cid: row for row, cid in enumerate(unique_cids)}

    outer_cluster = _cluster_keys(unique_cids, fp_bits, cutoff, seed)
    outer_of = _fold_of(unique_cids, outer_cluster, n_outer, seed)
    outer = np.array([outer_of[cid] for cid in unique_cids], dtype=np.int16)
    cluster = np.array([outer_cluster[cid] for cid in unique_cids], dtype=np.int32)

    weight = Counter(compound_ids.tolist())

    inner = np.full((len(unique_cids), n_outer), TEST, dtype=np.int16)
    for outer_i in range(n_outer):
        train_cids = [cid for cid in unique_cids if outer_of[cid] != outer_i]
        fold_seed = seed + 1 + outer_i
        train_rows = np.array([row_of_cid[cid] for cid in train_cids])
        inner_cluster = _cluster_keys(train_cids, fp_bits[train_rows], cutoff, fold_seed)

        calibration_cids, fit_cids = _carve_calibration(
            train_cids, inner_cluster, weight, calibration_fraction, fold_seed
        )
        inner[[row_of_cid[cid] for cid in calibration_cids], outer_i] = CALIBRATION
        inner_of = _fold_of(fit_cids, inner_cluster, n_inner, fold_seed)
        inner[[row_of_cid[cid] for cid in fit_cids], outer_i] = np.array(
            [inner_of[cid] for cid in fit_cids], dtype=np.int16
        )

    return SplitAssignment(
        task=task,
        compound_ids=unique_cids,
        cluster=cluster,
        outer=outer,
        inner=inner,
        n_outer=n_outer,
        n_inner=n_inner,
        seed=seed,
        cutoff=cutoff,
        calibration_fraction=calibration_fraction,
        rdkit_version=rdkit.__version__,
        compound_digest=_compound_set_digest(unique_cids),
    )


# --------------------------------------------------------------------------- IO

# Bumped when the file layout changes in a way older readers cannot parse.
SPLIT_SCHEMA_VERSION = 2
_HEADER_FIELDS = (
    "schema_version",
    "task",
    "grouping",
    "fingerprint",
    "cutoff",
    "n_outer",
    "n_inner",
    "calibration_fraction",
    "seed",
    "n_compounds",
    "rdkit_version",
    "compound_digest",
)


def write_splits(assignment: SplitAssignment, path: str | Path) -> Path:
    """Write a split to CSV: a ``#`` header of build settings, a column header, then rows.

    Nothing time-varying is recorded, so regenerating from identical inputs
    reproduces the file byte for byte.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task": assignment.task,
        "grouping": "butina",
        "fingerprint": f"morgan_bits_{MORGAN_BITS}_r{MORGAN_RADIUS}_largest_fragment",
        "cutoff": assignment.cutoff,
        "n_outer": assignment.n_outer,
        "n_inner": assignment.n_inner,
        "calibration_fraction": assignment.calibration_fraction,
        "seed": assignment.seed,
        "n_compounds": len(assignment.compound_ids),
        "rdkit_version": assignment.rdkit_version,
        "compound_digest": assignment.compound_digest,
    }
    with path.open("w", newline="") as handle:
        for key in _HEADER_FIELDS:
            handle.write(f"# {key}: {header[key]}\n")
        writer = csv.writer(handle)
        writer.writerow(
            [
                "compound_id",
                "cluster",
                "outer",
                *[f"inner_{k}" for k in range(assignment.n_outer)],
            ]
        )
        for i, cid in enumerate(assignment.compound_ids):
            writer.writerow(
                [
                    cid,
                    int(assignment.cluster[i]),
                    int(assignment.outer[i]),
                    *assignment.inner[i].tolist(),
                ]
            )
    return path


def read_splits(path: str | Path) -> SplitAssignment:
    """Read a split file.

    Raises:
        ValueError: on a schema version this reader does not know, or if the
            stored compound digest disagrees with the rows in the file
            (truncation or hand-editing).
    """
    path = Path(path)
    header: dict[str, str] = {}
    rows: list[list[str]] = []
    with path.open(newline="") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            header[key.strip()] = value.strip()
        reader = csv.reader(handle)  # the loop above consumed the column header
        rows = [row for row in reader if row]

    version = int(header["schema_version"])
    if version != SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: split schema version {version}, this reader speaks {SPLIT_SCHEMA_VERSION}"
        )
    n_outer = int(header["n_outer"])
    compound_ids = [row[0] for row in rows]
    cluster = np.array([int(row[1]) for row in rows], dtype=np.int32)
    outer = np.array([int(row[2]) for row in rows], dtype=np.int16)
    inner = np.array([[int(v) for v in row[3 : 3 + n_outer]] for row in rows], dtype=np.int16)

    digest = _compound_set_digest(compound_ids)
    if digest != header["compound_digest"]:
        raise ValueError(
            f"{path}: rows do not match the stored compound digest; the file has been "
            f"truncated or edited."
        )
    return SplitAssignment(
        task=header["task"],
        compound_ids=compound_ids,
        cluster=cluster,
        outer=outer,
        inner=inner,
        n_outer=n_outer,
        n_inner=int(header["n_inner"]),
        seed=int(header["seed"]),
        cutoff=float(header["cutoff"]),
        calibration_fraction=float(header["calibration_fraction"]),
        rdkit_version=header["rdkit_version"],
        compound_digest=header["compound_digest"],
    )


def _cli(
    task: str,
    out_dir: Path,
    seed: int,
    data_root: Path = Path("data/tdc_admet"),
    cache_root: Path = Path("data/features"),
    n_outer: int = 5,
    n_inner: int = 5,
    cutoff: float = BUTINA_CUTOFF,
) -> None:
    """Generate and write nested-CV splits (use ``all`` for every task)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from admet_uq import data as data_mod
    from admet_uq import featurize as featurize_mod

    for name in list(data_mod.TASKS) if task == "all" else [task]:
        # Split on exactly the rows that will be modelled: featurization drops
        # rows it cannot represent.
        feats = featurize_mod.featurize_task(name, data_root, cache_root)
        assignment = assign_nested_cv(
            feats["compound_id"],
            feats["smiles"],
            task=name,
            n_outer=n_outer,
            n_inner=n_inner,
            seed=seed,
            cutoff=cutoff,
        )
        path = write_splits(assignment, out_dir / f"{name}.csv")
        sizes = np.bincount(assignment.cluster)
        folds = assignment.to_folds(feats["compound_id"])
        calibration = [len(fold.calibration_idx) for fold in folds]
        realised = [len(f.calibration_idx) / len(f.train_idx) for f in folds]
        log.info(
            "[%s] %d rows / %d compounds / %d clusters (largest %.1f%%); "
            "calib n %d-%d (%.1f-%.1f%% of outer_train) -> %s",
            name,
            len(feats["compound_id"]),
            len(assignment.compound_ids),
            len(sizes),
            100 * sizes.max() / len(assignment.compound_ids),
            min(calibration),
            max(calibration),
            100 * min(realised),
            100 * max(realised),
            path,
        )


if __name__ == "__main__":
    import tyro

    tyro.cli(_cli)
