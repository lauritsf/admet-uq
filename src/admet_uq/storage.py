"""Read and write per-cell results.

A completed cell contains metric curves, prediction trajectories, evaluation
rows, and metadata. Predictions are stored in original target units; declared
scale outputs remain in model space.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# The current layout includes named prediction outputs and selection rules.
CELL_SCHEMA_VERSION = 3

CURVES_FILE = "curves.parquet"
PREDICTIONS_FILE = "predictions.parquet"
EVAL_ROWS_FILE = "eval_rows.parquet"
METADATA_FILE = "cell.json"

PARTIAL_DIR = ".partial"
PARTIAL_MANIFEST = "manifest.json"

_COMPRESSION = "zstd"
# Larger row groups compress a little better; this size bounds the writer's
# resident set.
ROW_GROUP_BYTES = 64 * 1024 * 1024


def cell_dir(results_root: str | Path, task: str, head: str, outer: int) -> Path:
    """Directory holding one cell's artifacts."""
    return Path(results_root) / task / head / f"outer{outer}"


@dataclass(frozen=True)
class EvalRows:
    """The rows a block of prediction vectors is aligned to.

    ``row_index`` are positions into the task-level feature arrays, so a stored
    prediction can always be traced back to its molecule. ``y_true`` is in the
    task's original units.
    """

    stage: str  # "inner" | "final"
    inner_fold: int  # -1 for the final refit
    eval_set: str  # "val" | "calib" | "test"
    row_index: np.ndarray
    compound_id: np.ndarray
    y_true: np.ndarray


@dataclass(frozen=True)
class PredictionBlock:
    """Predictions at every iteration for one fit on one evaluation set.

    ``preds`` is ``(n_iterations, n_rows)``: row ``i`` is the prediction vector
    after ``first_iteration + i`` iterations, aligned to the :class:`EvalRows`
    entry with the same ``(stage, inner_fold, eval_set)``. A fit that produces
    its trajectory one iteration at a time writes several blocks, each carrying
    the iteration its first row belongs to.

    ``output`` names which value of the predictive summary the block holds, and
    iterations are numbered within it. A fit predicting a single value per row
    leaves it at ``mean``.

    Cast to float32 on construction, which is the stored column type.
    """

    stage: str
    config_id: str
    inner_fold: int
    eval_set: str
    preds: np.ndarray
    first_iteration: int = 1
    output: str = "mean"

    def __post_init__(self):
        """Enforce the float32 invariant on construction."""
        if self.preds.dtype != np.float32:
            object.__setattr__(self, "preds", np.ascontiguousarray(self.preds, dtype=np.float32))


def write_curves(path: str | Path, records: list[dict]) -> Path:
    """Write the ``(selection, config, inner fold, iteration) -> metric`` curve table.

    Each record carries ``selection``, ``config_id``, ``inner_fold`` and a
    ``curve`` array of that rule's per-iteration values. A cell whose arm was
    read off another head's sweep has no curves of its own and writes the table
    empty, so every cell holds the same four files.
    """
    selection: list[str] = []
    config_id: list[str] = []
    inner_fold: list[int] = []
    iteration: list[np.ndarray] = [np.empty(0, dtype=np.int32)]
    value: list[np.ndarray] = [np.empty(0, dtype=np.float64)]
    for record in records:
        curve = np.asarray(record["curve"], dtype=np.float64)
        selection.extend([record["selection"]] * len(curve))
        config_id.extend([record["config_id"]] * len(curve))
        inner_fold.extend([record["inner_fold"]] * len(curve))
        iteration.append(np.arange(1, len(curve) + 1, dtype=np.int32))
        value.append(curve)

    table = pa.table(
        {
            "selection": pa.array(selection, type=pa.string()),
            "config_id": pa.array(config_id, type=pa.string()),
            "inner_fold": pa.array(inner_fold, type=pa.int16()),
            "iteration": pa.array(np.concatenate(iteration)),
            "metric": pa.array(np.concatenate(value)),
        }
    )
    return _write(table, path)


def _block_table(block: PredictionBlock) -> pa.Table:
    """One block as an arrow table: one row per ``(fit, iteration)``.

    The vectors go into a single ``list<float>`` column built directly from the
    flat ``(n_iterations * n_rows)`` buffer, so nothing is materialised as Python
    lists on the way to parquet.
    """
    preds = block.preds
    n_iterations, n_rows = preds.shape
    offsets = np.arange(0, n_iterations * n_rows + 1, n_rows, dtype=np.int32)
    return pa.table(
        {
            "stage": pa.array([block.stage] * n_iterations, type=pa.string()),
            "config_id": pa.array([block.config_id] * n_iterations, type=pa.string()),
            "inner_fold": pa.array(np.full(n_iterations, block.inner_fold, dtype=np.int16)),
            "eval_set": pa.array([block.eval_set] * n_iterations, type=pa.string()),
            "output": pa.array([block.output] * n_iterations, type=pa.string()),
            "iteration": pa.array(
                np.arange(
                    block.first_iteration,
                    block.first_iteration + n_iterations,
                    dtype=np.int32,
                )
            ),
            "pred": pa.ListArray.from_arrays(pa.array(offsets), pa.array(preds.reshape(-1))),
        }
    )


class PredictionWriter:
    """Stream prediction blocks to a Parquet file with bounded buffering."""

    def __init__(self, path: str | Path, row_group_bytes: int = ROW_GROUP_BYTES):
        """Target the final parquet path; writing goes to a sibling ``.tmp`` first."""
        self._path = Path(path)
        self._tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        self._writer: pq.ParquetWriter | None = None
        self._row_group_bytes = row_group_bytes
        self._buffer: list[pa.Table] = []
        self._buffered = 0

    def __enter__(self) -> PredictionWriter:
        """Ensure the cell directory exists; the file opens on the first write."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, block: PredictionBlock) -> None:
        """Buffer one block, flushing a row group once the buffer is large enough."""
        self.write_table(_block_table(block), block.preds.nbytes)

    def write_table(self, table: pa.Table, nbytes: int | None = None) -> None:
        """Buffer rows already in the stored schema, which is what compaction replays.

        ``nbytes`` is what the rows occupy unpacked, since the row-group size is
        chosen against the resident set rather than against the encoded table.
        """
        self._buffer.append(table)
        self._buffered += table.nbytes if nbytes is None else nbytes
        if self._buffered >= self._row_group_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.concat_tables(self._buffer) if len(self._buffer) > 1 else self._buffer[0]
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._tmp, table.schema, compression=_COMPRESSION)
        self._writer.write_table(table, row_group_size=len(table))
        self._buffer.clear()
        self._buffered = 0

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Close and rename into place; on failure leave no partial artifact."""
        if exc_type is None:
            self._flush()
        self._buffer.clear()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if exc_type is None:
            self._tmp.replace(self._path)
        elif self._tmp.exists():
            self._tmp.unlink()
        return False


@cache
def source_digest() -> str:
    """SHA-256 over this package's sources.

    A resumed cell must compute what an uninterrupted one would, so a partial
    store is keyed on the code that produced it as well as on the inputs.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


class PartialCell:
    """Store completed configurations until a cell is finalized.

    Matching provenance allows a run to resume at configuration boundaries.
    """

    def __init__(self, directory: str | Path, provenance: dict):
        """Hold the store's directory and what its shards must have been built from."""
        self._dir = Path(directory)
        self._provenance = provenance
        self._buffer: list[pa.Table] = []
        self._configs: list[str] = []
        self._refits: list[str] = []

    @property
    def configs(self) -> list[str]:
        """Config ids sealed here, in the order they were finished."""
        return list(self._configs)

    def resume(self) -> list[str]:
        """Adopt a matching store and return the config ids it already holds.

        A store built from anything else is removed, so the caller always starts
        from a directory that agrees with its own inputs.
        """
        manifest = self._dir / PARTIAL_MANIFEST
        held = json.loads(manifest.read_text()) if manifest.is_file() else None
        if held is None or held.get("provenance") != self._provenance:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest()
            return []
        self._configs = [name for name in held["configs"] if self._shard(name, "curves").is_file()]
        for stale in self._dir.glob("refit__*"):
            stale.unlink()
        self._write_manifest()
        return list(self._configs)

    def curve_records(self) -> list[dict]:
        """The curve rows of every sealed config, as :func:`write_curves` takes them."""
        records = []
        for config_id in self._configs:
            table = pq.read_table(self._shard(config_id, "curves")).to_pandas()
            table = table.sort_values("iteration")
            for key, rows in table.groupby(["selection", "config_id", "inner_fold"], sort=False):
                records.append(
                    {
                        "selection": key[0],
                        "config_id": key[1],
                        "inner_fold": int(key[2]),
                        "curve": rows["metric"].to_numpy(),
                    }
                )
        return records

    def write(self, block: PredictionBlock) -> None:
        """Take one block into the shard being built, which is what a sink calls."""
        self._buffer.append(_block_table(block))

    def seal(self, config_id: str, curve_records: list[dict]) -> None:
        """Store one finished config, naming it in the manifest last.

        Sealing a config the store already holds replaces its shards and leaves
        the manifest one entry long, so a config recomputed because a sibling
        store lacked it does not compact into duplicate rows.
        """
        self._store_buffer(config_id)
        write_curves(self._shard(config_id, "curves"), curve_records)
        if config_id not in self._configs:
            self._configs.append(config_id)
        self._write_manifest()

    def seal_refit(self, config_id: str) -> None:
        """Store one finished refit, which a later attempt drops rather than reuses."""
        name = f"refit__{config_id}"
        self._store_buffer(name)
        self._refits.append(name)

    def compact(self, path: str | Path) -> Path:
        """Stream every shard into the cell's predictions file, in the order produced.

        Shards are replayed a row group at a time, and :func:`_row_group_rows`
        holds a group to ``ROW_GROUP_BYTES``, so the resident set stays at that
        rather than at one config.
        """
        with PredictionWriter(path) as writer:
            for name in self._configs + self._refits:
                shard = self._shard(name, "preds")
                if not shard.is_file():
                    continue
                for batch in pq.ParquetFile(shard).iter_batches():
                    writer.write_table(pa.Table.from_batches([batch]))
        return Path(path)

    def _store_buffer(self, name: str) -> None:
        """Write the buffered blocks as this shard's predictions, then drop them."""
        if self._buffer:
            _write(pa.concat_tables(self._buffer), self._shard(name, "preds"))
            self._buffer.clear()

    def _shard(self, name: str, kind: str) -> Path:
        return self._dir / f"{name}__{kind}.parquet"

    def _write_manifest(self) -> None:
        write_metadata(
            self._dir / PARTIAL_MANIFEST,
            {"provenance": self._provenance, "configs": self._configs},
        )


def discard_partial(cell_dir: str | Path) -> None:
    """Remove a completed cell's partial store."""
    shutil.rmtree(Path(cell_dir) / PARTIAL_DIR, ignore_errors=True)


def write_eval_rows(path: str | Path, eval_rows: list[EvalRows]) -> Path:
    """Write the alignment table: one row per evaluated observation."""
    tables = [
        pa.table(
            {
                "stage": pa.array([rows.stage] * len(rows.row_index), type=pa.string()),
                "inner_fold": pa.array(
                    np.full(len(rows.row_index), rows.inner_fold, dtype=np.int16)
                ),
                "eval_set": pa.array([rows.eval_set] * len(rows.row_index), type=pa.string()),
                "row_index": pa.array(np.asarray(rows.row_index, dtype=np.int32)),
                "compound_id": pa.array([str(cid) for cid in rows.compound_id], type=pa.string()),
                "y_true": pa.array(np.asarray(rows.y_true, dtype=np.float64)),
            }
        )
        for rows in eval_rows
    ]
    return _write(pa.concat_tables(tables), path)


def write_metadata(path: str | Path, metadata: dict) -> Path:
    """Write ``cell.json`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_jsonable) + "\n")
    tmp.replace(path)
    return path


def _jsonable(value):
    """Coerce numpy scalars/arrays that ``json`` cannot serialise on its own."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def _row_group_rows(table: pa.Table) -> int:
    """Rows holding about ``ROW_GROUP_BYTES`` unpacked.

    A reader takes a whole row group at once, so this is what bounds the resident
    set of anything streaming the file back.
    """
    if table.num_rows == 0:
        return 1
    per_row = max(1, table.nbytes // table.num_rows)
    return max(1, min(table.num_rows, ROW_GROUP_BYTES // per_row))


def _write(table: pa.Table, path: str | Path) -> Path:
    """Write a Parquet file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression=_COMPRESSION, row_group_size=_row_group_rows(table))
    tmp.replace(path)
    return path


def read_curves(path: str | Path):
    """Read a cell's curve table as a pandas DataFrame."""
    return pq.read_table(path).to_pandas()


def read_predictions(path: str | Path, **filters):
    """Read prediction rows, optionally filtered on the key columns.

    Filters are exact matches on ``stage`` / ``config_id`` / ``inner_fold`` /
    ``eval_set`` / ``output`` / ``iteration``, pushed down to parquet so a single
    iteration can be pulled out of an 80 GB benchmark without reading the rest.
    """
    expression = None
    import pyarrow.dataset as ds

    for column, wanted in filters.items():
        clause = ds.field(column) == wanted
        expression = clause if expression is None else (expression & clause)
    return pq.read_table(path, filters=expression).to_pandas()
