"""Nested cross-validation for hyperparameter selection and outer-fold evaluation.

Each cell evaluates all configurations on the inner folds, selects a
configuration and iteration from the mean validation curve, then refits on the
outer training data and records calibration and test predictions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from admet_uq.curves import EvalSink, GroupSink, pad_to
from admet_uq.data import primary_metric, target_transform, transform_target
from admet_uq.heads import SHARED_SELECTION, get_head, group_fit
from admet_uq.metrics import accuracy_metrics
from admet_uq.splits import OuterFold, SplitAssignment
from admet_uq.storage import (
    CELL_SCHEMA_VERSION,
    PARTIAL_DIR,
    PREDICTIONS_FILE,
    EvalRows,
    PartialCell,
    source_digest,
)

log = logging.getLogger(__name__)


def _config_digest(configs: list[dict]) -> str:
    """SHA-256 over the config list, so a resumed cell sweeps the same search space."""
    payload = json.dumps(configs, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _fold_seeds(seed: int, n_inner: int) -> list[int]:
    """Derive one seed per inner fold and one for refitting.

    The same fold seed is reused across configurations to preserve paired
    comparisons.
    """
    return [int(value) for value in np.random.SeedSequence(seed).generate_state(n_inner + 1)]


def member_seeds(seed: int, n_members: int) -> list[int]:
    """Derive one independent seed per ensemble member."""
    children = np.random.SeedSequence(seed).spawn(n_members)
    return [int(child.generate_state(1)[0]) for child in children]


def _select(
    configs: list[dict], curves: np.ndarray, direction: str, task: str, outer: int
) -> tuple[dict, int, np.ndarray, int]:
    """Pick a config and iteration from the mean of each config's inner-fold curves.

    Folds that scored NaN are skipped.

    Returns:
        ``(selected_config, selected_iteration, config_scores, selected_index)``.
    """
    with np.errstate(invalid="ignore"):
        mean_curves = np.nanmean(curves, axis=1)
    if np.isnan(mean_curves).all():
        raise ValueError(f"[{task}/outer{outer}] every inner curve is NaN; cannot select a config")
    pick = np.nanargmax if direction == "maximize" else np.nanargmin
    best_iteration_per_config = np.array([pick(curve) for curve in mean_curves]) + 1
    config_scores = mean_curves[np.arange(len(configs)), best_iteration_per_config - 1]
    selected = int(pick(config_scores))
    return (
        configs[selected],
        int(best_iteration_per_config[selected]),
        config_scores,
        selected,
    )


@dataclass(frozen=True)
class CellOutputs:
    """The small artifacts of one cell; predictions are streamed out separately."""

    metadata: dict
    curve_records: list[dict]
    eval_rows: list[EvalRows]


def _fit_call(names: Sequence[str]):
    """What trains one network for these heads, and how their sinks are built.

    A single head trains through its own ``fit`` and takes plain sinks, so a run
    of one head computes what it computed before any group existed. Several take
    the backbone's combined fit and a :class:`~admet_uq.curves.GroupSink` per
    evaluation set.
    """
    if len(names) == 1:
        head = get_head(names[0])
        return head.fit, lambda sinks: list(sinks.values())
    return group_fit(names), lambda sinks: [GroupSink(sinks)]


def run_cells(
    task: str,
    heads: Sequence[str],
    outer: int,
    features: dict[str, np.ndarray],
    assignment: SplitAssignment,
    folds: list[OuterFold],
    configs: list[dict],
    cell_dirs: Mapping[str, str | Path],
    *,
    self_selecting: Sequence[str] | None = None,
    n_iterations: int | None = None,
    n_jobs: int = 0,
    seed: int = 0,
    store_inner_predictions: bool = True,
) -> dict[str, CellOutputs]:
    """Run one ``(task, outer fold)`` over heads that share a trained network.

    Args:
        task: TDC ADMET task name.
        heads: head names, the first being the deterministic one whose selection
            the rest deploy. One name runs that head on its own.
        outer: outer fold index.
        features: the ``featurize_task`` dict (``X``, ``y``, ``compound_id``).
        assignment: the persisted split; its calibration fraction and compound
            digest go into the cell metadata.
        folds: ``assignment.to_folds(...)``.
        configs: the presampled config list.
        cell_dirs: where each head's artifacts are written.
        self_selecting: heads whose own curves the sweep records and which
            therefore choose their own config and iteration; ``None`` takes all
            of them. The first head is always among them.
        n_iterations: training iterations per fit, recorded at every step;
            ``None`` takes the head's default.
        n_jobs: worker threads (0 = all).
        seed: model seed.
        store_inner_predictions: write the per-iteration inner-validation
            predictions.
    """
    specs = {name: get_head(name) for name in heads}
    primary_head = heads[0]
    swept = tuple(heads if self_selecting is None else self_selecting)
    if swept[0] != primary_head:
        raise ValueError(f"{primary_head!r} selects for the group and must sweep its own curves")
    if not set(swept) <= set(heads):
        raise ValueError(f"self_selecting {swept} names heads outside {tuple(heads)}")

    if n_iterations is None:
        n_iterations = specs[primary_head].DEFAULT_ITERATIONS
    started = time.time()
    fold_seeds = _fold_seeds(seed, len(folds[outer].inner))
    x_all, y_raw, compound_id = features["X"], features["y"], features["compound_id"]
    y_fit_space = transform_target(task, y_raw)
    fold = folds[outer]

    calibration_idx, fit_idx, inner_folds = fold.calibration_idx, fold.fit_idx, fold.inner
    log.info(
        "[%s/%s/outer%d] train %d -> fit %d + calib %d (%.1f%%); test %d;"
        " inner val %s; sweeping %s",
        task,
        "+".join(heads),
        outer,
        len(fold.train_idx),
        len(fit_idx),
        len(calibration_idx),
        100 * len(calibration_idx) / len(fold.train_idx),
        len(fold.test_idx),
        [len(val) for _, val in inner_folds],
        "+".join(swept),
    )

    eval_rows = [
        EvalRows("inner", k, "val", val, compound_id[val], y_raw[val])
        for k, (_, val) in enumerate(inner_folds)
    ]
    objectives = {
        name: [specs[name].objectives(task, rows) for rows in eval_rows] for name in swept
    }
    rules = {name: list(objectives[name][0]) for name in swept}
    curves = {
        name: {
            rule: np.full((len(configs), len(inner_folds), n_iterations), np.nan)
            for rule in rules[name]
        }
        for name in swept
    }

    provenance = {
        "task": task,
        "heads": list(heads),
        "self_selecting": list(swept),
        "outer": outer,
        "n_iterations": n_iterations,
        "seed": seed,
        "fold_seeds": fold_seeds,
        "config_digest": _config_digest(configs),
        "compound_digest": assignment.compound_digest,
        "store_inner_predictions": store_inner_predictions,
        "schema_version": CELL_SCHEMA_VERSION,
        "source_digest": source_digest(),
    }
    writers = {
        name: PartialCell(Path(cell_dirs[name]) / PARTIAL_DIR, provenance | {"head": name})
        for name in heads
    }
    # A config is complete only when every swept cell contains it.
    sealed = {name: set(writers[name].resume()) for name in heads}
    done = set.intersection(*(sealed[name] for name in swept))
    curve_records = {name: writers[name].curve_records() if name in swept else [] for name in heads}
    position = {config["config_id"]: i for i, config in enumerate(configs)}
    for name in swept:
        for record in curve_records[name]:
            if record["config_id"] in done:
                curves[name][record["selection"]][
                    position[record["config_id"]], record["inner_fold"]
                ] = pad_to(record["curve"], n_iterations)
        curve_records[name] = [r for r in curve_records[name] if r["config_id"] in done]
    if done:
        log.info(
            "[%s/%s/outer%d] resuming: %d of %d configs already stored",
            task,
            primary_head,
            outer,
            len(done),
            len(configs),
        )

    sweep_fit, sweep_sinks = _fit_call(swept)
    for config_i, config in enumerate(configs):
        if config["config_id"] in done:
            continue
        fresh: dict[str, list[dict]] = {name: [] for name in swept}
        for fold_k, (inner_train, inner_val) in enumerate(inner_folds):
            # Sliced once and shared: the group predicts one block of rows, and
            # a copy per head would hold the features as many times over.
            x_val = x_all[inner_val]
            sinks = {
                specs[name].UQ: EvalSink(
                    task,
                    eval_rows[fold_k],
                    config["config_id"],
                    x_val,
                    writer=writers[name] if store_inner_predictions else None,
                    scorers={
                        rule: objective.score
                        for rule, objective in objectives[name][fold_k].items()
                    },
                    scale_outputs=specs[name].SCALE_OUTPUTS,
                )
                for name in swept
            }
            sweep_fit(
                task,
                x_all[inner_train],
                y_fit_space[inner_train],
                config,
                sweep_sinks(sinks),
                seed=fold_seeds[fold_k],
                n_jobs=n_jobs,
                n_iterations=n_iterations,
            )
            for name in swept:
                sink = sinks[specs[name].UQ]
                for rule in rules[name]:
                    curves[name][rule][config_i, fold_k] = pad_to(sink.curve(rule), n_iterations)
                    fresh[name].append(
                        {
                            "selection": rule,
                            "config_id": config["config_id"],
                            "inner_fold": fold_k,
                            "curve": curves[name][rule][config_i, fold_k],
                        }
                    )
        for name in swept:
            writers[name].seal(config["config_id"], fresh[name])
            curve_records[name].extend(fresh[name])
        log.info(
            "[%s/%s/outer%d] config %d/%d %s done (%.0fs elapsed)",
            task,
            primary_head,
            outer,
            config_i + 1,
            len(configs),
            config["config_id"],
            time.time() - started,
        )

    selections: dict[str, dict[str, dict]] = {name: {} for name in heads}
    for name in swept:
        for rule in rules[name]:
            objective = objectives[name][0][rule]
            chosen, iteration, config_scores, index = _select(
                configs, curves[name][rule], objective.direction, task, outer
            )
            selections[name][rule] = {
                "selection": rule,
                "selection_metric": objective.name,
                "selection_direction": objective.direction,
                "selected_config_id": chosen["config_id"],
                "selected_config": {k: v for k, v in chosen.items() if k != "config_id"},
                "selected_iteration": iteration,
                "mean_inner_score": float(config_scores[index]),
            }
            log.info(
                "[%s/%s/outer%d] %s selected %s at iteration %d (mean inner %s = %.4f)",
                task,
                name,
                outer,
                rule,
                chosen["config_id"],
                iteration,
                objective.name,
                config_scores[index],
            )

    # Secondary heads deploy the configuration selected by the primary head.
    headline = selections[primary_head][rules[primary_head][0]]
    for name in heads[1:]:
        selections[name][SHARED_SELECTION] = {
            key: headline[key]
            for key in ("selected_config_id", "selected_config", "selected_iteration")
        } | {
            "selection": SHARED_SELECTION,
            "selection_metric": headline["selection_metric"],
            "selection_direction": headline["selection_direction"],
            "mean_inner_score": headline["mean_inner_score"],
        }

    refit_rows = [
        EvalRows("final", -1, name, idx, compound_id[idx], y_raw[idx])
        for name, idx in (("calib", calibration_idx), ("test", fold.test_idx))
    ]
    eval_rows = eval_rows + refit_rows

    # Rules landing on one config share its refit: the trajectory is recorded
    # whole, so they differ only in the iteration each reads it at. The first
    # pair to claim a config owns its checkpoint.
    owners: dict[str, tuple[str, str]] = {}
    claims: dict[str, list[tuple[str, str]]] = {}
    for name in heads:
        for rule in _primary_first(selections[name], name, primary_head, rules):
            config_id = selections[name][rule]["selected_config_id"]
            owners.setdefault(config_id, (name, rule))
            claims.setdefault(config_id, []).append((name, rule))
    configs_by_id = {config["config_id"]: config for config in configs}

    refit_fit, refit_sinks_of = _fit_call(heads)
    # Sliced once for every refit, and shared across the heads of each, as the
    # sweep's blocks are.
    refit_features = [x_all[rows.row_index] for rows in refit_rows]
    head_fit: dict = {}
    for config_id, group in claims.items():
        owner_head, owner_rule = owners[config_id]
        artifacts = Path(cell_dirs[owner_head])
        if (owner_head, owner_rule) != (primary_head, rules[primary_head][0]):
            artifacts = artifacts / f"refit_{owner_rule}"
        wanted = {
            name: sorted(
                {
                    selections[name][rule]["selected_iteration"]
                    for other, rule in group
                    if other == name
                }
            )
            for name in heads
        }
        sinks_per_rows = [
            {
                specs[name].UQ: EvalSink(
                    task,
                    rows,
                    config_id,
                    block,
                    writer=writers[name],
                    scale_outputs=specs[name].SCALE_OUTPUTS,
                    keep_iterations=wanted[name] if rows.eval_set == "test" else (),
                    keep_output=specs[name].POINT_OUTPUT,
                )
                for name in heads
            }
            for rows, block in zip(refit_rows, refit_features, strict=True)
        ]
        refit = refit_fit(
            task,
            x_all[fit_idx],
            y_fit_space[fit_idx],
            configs_by_id[config_id],
            [sink for sinks in sinks_per_rows for sink in refit_sinks_of(sinks)],
            seed=fold_seeds[-1],
            n_jobs=n_jobs,
            n_iterations=n_iterations,
            artifacts_dir=artifacts,
            checkpoint_at=selections[owner_head][owner_rule]["selected_iteration"],
        )
        for name in heads:
            writers[name].seal_refit(config_id)
        if (owner_head, owner_rule) == (primary_head, rules[primary_head][0]):
            head_fit = refit.metadata
        for name, rule in group:
            iteration = selections[name][rule]["selected_iteration"]
            # The refit can end below the selected iteration; the model is
            # unchanged past its last one, so clamping is the correct read.
            test_metrics = accuracy_metrics(
                task,
                y_raw[fold.test_idx],
                sinks_per_rows[-1][specs[name].UQ].kept[iteration],
                compound_id[fold.test_idx],
            )
            selections[name][rule] |= {
                "deployed_iteration": min(iteration, refit.n_iterations),
                "refit_n_iterations": refit.n_iterations,
                "refit_dir": str(artifacts.relative_to(Path(cell_dirs[owner_head]))),
                "refit_head": owner_head,
                "test_metrics": test_metrics,
            }
            log.info(
                "[%s/%s/outer%d] %s test %s = %.4f (%.0fs elapsed)",
                task,
                name,
                outer,
                rule,
                primary_metric(task)[0],
                test_metrics["primary"],
                time.time() - started,
            )

    outputs = {}
    for name in heads:
        writers[name].compact(Path(cell_dirs[name]) / PREDICTIONS_FILE)
        ordered = _primary_first(selections[name], name, primary_head, rules)
        spec = specs[name]
        metadata = {
            "task": task,
            "head": spec.HEAD,
            "uq": spec.UQ,
            "outputs": list(spec.OUTPUTS),
            "scale_outputs": list(spec.SCALE_OUTPUTS),
            "point_output": spec.POINT_OUTPUT,
            "outer": outer,
            "primary_metric": primary_metric(task)[0],
            "direction": primary_metric(task)[1],
            "target_transform": target_transform(task),
            "selections": {rule: selections[name][rule] for rule in ordered},
            **selections[name][ordered[0]],
            "head_fit": head_fit,
            "shared_sweep": list(heads) if len(heads) > 1 else None,
            "n_configs": len(configs),
            "n_iterations": n_iterations,
            "seed": seed,
            "fold_seeds": fold_seeds,
            "calibration_fraction_target": assignment.calibration_fraction,
            "calibration_fraction_realised": len(calibration_idx) / len(fold.train_idx),
            "n_rows": {
                "outer_train": len(fold.train_idx),
                "fit": len(fit_idx),
                "calibration": len(calibration_idx),
                "test": len(fold.test_idx),
                "inner_val": [len(val) for _, val in inner_folds],
            },
            "inner_predictions_stored": store_inner_predictions and name in swept,
            "compound_digest": assignment.compound_digest,
            "runtime_seconds": time.time() - started,
        }
        outputs[name] = CellOutputs(metadata, curve_records[name], eval_rows)
    return outputs


def _primary_first(
    selected: Mapping[str, dict], name: str, primary_head: str, rules: Mapping[str, list[str]]
) -> list[str]:
    """This head's rules, the one owning its cell directory first.

    The deterministic head owns its cell under the rule it was selected by; a
    head deploying that selection owns its cell under the shared rule, which is
    what its flat metadata fields describe.
    """
    first = rules[primary_head][0] if name == primary_head else SHARED_SELECTION
    return [first] + [rule for rule in selected if rule != first]


def run_cell(
    task: str,
    head: str,
    outer: int,
    features: dict[str, np.ndarray],
    assignment: SplitAssignment,
    folds: list[OuterFold],
    configs: list[dict],
    cell_dir: str | Path,
    **kwargs,
) -> CellOutputs:
    """Run one ``(task, head, outer fold)`` cell on its own.

    Takes and returns what :func:`run_cells` does for a single head.
    """
    return run_cells(
        task, [head], outer, features, assignment, folds, configs, {head: cell_dir}, **kwargs
    )[head]


def run_member_cells(
    task: str,
    heads: Sequence[str],
    outer: int,
    features: dict[str, np.ndarray],
    assignment: SplitAssignment,
    folds: list[OuterFold],
    config: dict,
    cell_dirs: Mapping[str, str | Path],
    source_selections: Mapping[str, dict],
    *,
    member: int,
    member_seed: int,
    n_iterations: int | None = None,
    n_jobs: int = 0,
    seed: int = 0,
) -> dict[str, CellOutputs]:
    """Refit one ensemble member and write a cell for each head.

    Args:
        task: TDC ADMET task name.
        heads: head names, the first owning the trained network's checkpoints.
        outer: outer fold index.
        features: the ``featurize_task`` dict.
        assignment: the persisted split.
        folds: ``assignment.to_folds(...)``.
        config: the config row the source cell selected.
        cell_dirs: where each head's artifacts are written.
        source_selections: per head, the selection block its source cell holds,
            which names the rule and the iteration this member inherits.
        member: index of this member within its ensemble.
        member_seed: the seed it trains under.
        n_iterations: training iterations, ``None`` taking the head's default.
        n_jobs: worker threads (0 = all).
        seed: the ensemble's base seed, recorded so a member names its family.
    """
    specs = {name: get_head(name) for name in heads}
    primary_head = heads[0]
    if n_iterations is None:
        n_iterations = specs[primary_head].DEFAULT_ITERATIONS
    started = time.time()
    x_all, y_raw, compound_id = features["X"], features["y"], features["compound_id"]
    y_fit_space = transform_target(task, y_raw)
    fold = folds[outer]
    calibration_idx, fit_idx = fold.calibration_idx, fold.fit_idx
    selected_iteration = int(source_selections[primary_head]["selected_iteration"])

    log.info(
        "[%s/%s/outer%d] member %d at %s iteration %d: fit %d, calib %d, test %d",
        task,
        "+".join(heads),
        outer,
        member,
        config["config_id"],
        selected_iteration,
        len(fit_idx),
        len(calibration_idx),
        len(fold.test_idx),
    )

    eval_rows = [
        EvalRows("final", -1, name, idx, compound_id[idx], y_raw[idx])
        for name, idx in (("calib", calibration_idx), ("test", fold.test_idx))
    ]
    provenance = {
        "task": task,
        "heads": list(heads),
        "outer": outer,
        "n_iterations": n_iterations,
        "seed": seed,
        "member": member,
        "member_seed": member_seed,
        "config_digest": _config_digest([config]),
        "compound_digest": assignment.compound_digest,
        "schema_version": CELL_SCHEMA_VERSION,
        "source_digest": source_digest(),
    }
    writers = {
        name: PartialCell(Path(cell_dirs[name]) / PARTIAL_DIR, provenance | {"head": name})
        for name in heads
    }
    for writer in writers.values():
        writer.resume()

    blocks = [x_all[rows.row_index] for rows in eval_rows]
    sinks_per_rows = [
        {
            specs[name].UQ: EvalSink(
                task,
                rows,
                config["config_id"],
                block,
                writer=writers[name],
                scale_outputs=specs[name].SCALE_OUTPUTS,
                keep_iterations=(selected_iteration,) if rows.eval_set == "test" else (),
                keep_output=specs[name].POINT_OUTPUT,
            )
            for name in heads
        }
        for rows, block in zip(eval_rows, blocks, strict=True)
    ]
    fit_call, sinks_of = _fit_call(heads)
    refit = fit_call(
        task,
        x_all[fit_idx],
        y_fit_space[fit_idx],
        config,
        [sink for sinks in sinks_per_rows for sink in sinks_of(sinks)],
        seed=member_seed,
        n_jobs=n_jobs,
        n_iterations=n_iterations,
        artifacts_dir=Path(cell_dirs[primary_head]),
        checkpoint_at=selected_iteration,
    )

    outputs = {}
    for name in heads:
        writers[name].seal_refit(config["config_id"])
        writers[name].compact(Path(cell_dirs[name]) / PREDICTIONS_FILE)
        # The member can end below the inherited iteration; the model is
        # unchanged past its last one, so clamping is the correct read.
        test_metrics = accuracy_metrics(
            task,
            y_raw[fold.test_idx],
            sinks_per_rows[-1][specs[name].UQ].kept[selected_iteration],
            compound_id[fold.test_idx],
        )
        inherited = dict(source_selections[name]) | {
            "deployed_iteration": min(selected_iteration, refit.n_iterations),
            "refit_n_iterations": refit.n_iterations,
            "refit_dir": ".",
            "refit_head": primary_head,
            "test_metrics": test_metrics,
        }
        spec = specs[name]
        metadata = {
            "task": task,
            "head": spec.HEAD,
            "uq": spec.UQ,
            "outputs": list(spec.OUTPUTS),
            "scale_outputs": list(spec.SCALE_OUTPUTS),
            "point_output": spec.POINT_OUTPUT,
            "outer": outer,
            "primary_metric": primary_metric(task)[0],
            "direction": primary_metric(task)[1],
            "target_transform": target_transform(task),
            "selections": {inherited["selection"]: inherited},
            **inherited,
            "head_fit": refit.metadata,
            "shared_sweep": list(heads) if len(heads) > 1 else None,
            "member": member,
            "member_seed": member_seed,
            "n_configs": 1,
            "n_iterations": n_iterations,
            "seed": seed,
            "calibration_fraction_target": assignment.calibration_fraction,
            "calibration_fraction_realised": len(calibration_idx) / len(fold.train_idx),
            "n_rows": {
                "outer_train": len(fold.train_idx),
                "fit": len(fit_idx),
                "calibration": len(calibration_idx),
                "test": len(fold.test_idx),
            },
            "inner_predictions_stored": False,
            "compound_digest": assignment.compound_digest,
            "runtime_seconds": time.time() - started,
        }
        outputs[name] = CellOutputs(metadata, [], eval_rows)
        log.info(
            "[%s/%s/outer%d] member %d test %s = %.4f (%.0fs elapsed)",
            task,
            name,
            outer,
            member,
            primary_metric(task)[0],
            test_metrics["primary"],
            time.time() - started,
        )
    return outputs
