"""
Subsampling-based cohort stability analysis for PROTRIDER.

This is not classical bootstrap (which samples with replacement). Each iteration
removes a random subset of samples, reruns the full OHT pipeline including
preprocessing, and aggregates how stable outlier calls and metrics are.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from protrider.config import ProtriderConfig
from protrider.datasets.covariates import read_annotation_file
from protrider.datasets.protein_intensities import read_protein_intensities

logger = logging.getLogger(__name__)

MIN_COHORT_SAMPLES_FOR_STABILITY = 30

STABILITY_METRIC_COLUMNS = [
    "sampleID",
    "proteinID",
    "PROTEIN_ZSCORE",
    "PROTEIN_PVALUE",
    "PROTEIN_PADJ",
    "PROTEIN_LOG2FC",
    "PROTEIN_FC",
    "PROTEIN_outlier",
]

_SAMPLE_ID_COLUMN_CANDIDATES = ("sample_ID", "sampleID", "sample_id")


def run_cohort_stability(
    config: ProtriderConfig,
    baseline_summary: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Run subsampling-based cohort stability analysis.

    Repeatedly removes a random subset of samples, reruns the full PROTRIDER
    OHT pipeline including preprocessing, and aggregates sample-protein metrics
    across successful iterations.

    Args:
        config: Validated run configuration with cohort_stability enabled.
        baseline_summary: Long-format summary from the full-cohort run.

    Returns:
        Aggregated stability summary DataFrame, or None if analysis was skipped.
    """
    sample_ids = _read_sample_ids(config)
    n_samples = len(sample_ids)

    if n_samples <= MIN_COHORT_SAMPLES_FOR_STABILITY:
        logger.warning(
            "Cohort stability analysis skipped: cohort has %d samples (need > %d).",
            n_samples,
            MIN_COHORT_SAMPLES_FOR_STABILITY,
        )
        return None

    n_drop = _compute_n_drop(n_samples, config)
    if n_drop < 1:
        logger.warning(
            "Cohort stability analysis skipped: configured drop fraction and "
            "minimum retained samples do not allow removing any samples.",
        )
        return None

    base_seed = _resolve_stability_base_seed(config)
    plan = _generate_subsample_plan(sample_ids, config, base_seed, n_drop)

    n_requested = config.cohort_stability_n_runs
    max_runtime_sec = (
        config.cohort_stability_max_runtime_min * 60.0
        if config.cohort_stability_max_runtime_min is not None
        else None
    )

    logger.info(
        "Starting cohort stability analysis: %d iterations, %d samples, "
        "drop %d per iteration, base_seed=%s.",
        n_requested,
        n_samples,
        n_drop,
        base_seed,
    )

    iteration_summaries: List[pd.DataFrame] = []
    start_time = time.monotonic()
    completed = 0
    tmp_root = Path(config.out_dir) / "_cohort_stability_tmp"

    for iteration_index, retained_ids in enumerate(plan):
        if iteration_index >= n_requested:
            break

        iteration_seed = base_seed + iteration_index
        tmp_dir = tmp_root / f"iter_{iteration_index:04d}"

        try:
            logger.info(
                "Cohort stability iteration %d/%d (%d samples retained).",
                iteration_index + 1,
                n_requested,
                len(retained_ids),
            )
            iter_summary = _run_single_stability_iteration(
                config=config,
                retained_sample_ids=retained_ids,
                iteration_seed=iteration_seed,
                tmp_dir=tmp_dir,
            )
            iteration_summaries.append(iter_summary)
            completed += 1
        except Exception:
            logger.exception(
                "Cohort stability iteration %d failed.", iteration_index
            )
            raise
        finally:
            if not config.cohort_stability_save_iteration_files and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if (
            max_runtime_sec is not None
            and completed >= config.cohort_stability_min_runs
            and (time.monotonic() - start_time) >= max_runtime_sec
        ):
            logger.info(
                "Stopping cohort stability after %d iterations (runtime budget).",
                completed,
            )
            break

    if not iteration_summaries:
        logger.warning("Cohort stability analysis produced no successful iterations.")
        return None

    if not config.cohort_stability_save_iteration_files and tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    summary = _summarize_stability(
        iteration_summaries=iteration_summaries,
        baseline_summary=baseline_summary,
        n_runs_requested=n_requested,
    )
    logger.info(
        "Finished cohort stability: %d/%d iterations, summary shape %s.",
        len(iteration_summaries),
        n_requested,
        summary.shape,
    )
    return summary


def _read_sample_ids(config: ProtriderConfig) -> List[str]:
    """Read sample IDs from the configured intensity file without preprocessing."""
    data = read_protein_intensities(
        config.input_intensities,
        config.index_col,
        config.input_format,
    )
    return data.index.astype(str).tolist()


def _compute_n_drop(n_samples: int, config: ProtriderConfig) -> int:
    n_drop = round(n_samples * config.cohort_stability_drop_fraction)
    n_drop = max(1, n_drop)
    n_drop = min(n_drop, n_samples - config.cohort_stability_min_samples)
    return n_drop


def _resolve_stability_base_seed(config: ProtriderConfig) -> int:
    if config.cohort_stability_seed is not None:
        return int(config.cohort_stability_seed)
    if config.seed is not None:
        return int(config.seed)
    generated = int(np.random.default_rng().integers(0, 2**31 - 1))
    logger.info(
        "cohort_stability_seed and seed are unset; using generated seed %d.",
        generated,
    )
    return generated


def _generate_subsample_plan(
    sample_ids: List[str],
    config: ProtriderConfig,
    base_seed: int,
    n_drop: int,
) -> List[List[str]]:
    """Deterministic retained-sample lists for each stability iteration."""
    plan: List[List[str]] = []
    ids = list(sample_ids)
    for iteration_index in range(config.cohort_stability_n_runs):
        rng = np.random.default_rng(base_seed + iteration_index)
        drop_idx = rng.choice(len(ids), size=n_drop, replace=False)
        drop_set = {ids[i] for i in drop_idx}
        plan.append([sid for sid in ids if sid not in drop_set])
    return plan


def _subset_input_files(
    config: ProtriderConfig,
    retained_sample_ids: List[str],
    tmp_dir: Path,
) -> tuple[str, Optional[str]]:
    """Write temporary subset intensity and annotation files."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src = Path(config.input_intensities)
    suffix = src.suffix if src.suffix else ".tsv"
    sep = "\t" if suffix == ".tsv" else ","

    data = read_protein_intensities(
        config.input_intensities,
        config.index_col,
        config.input_format,
    )
    missing = set(retained_sample_ids) - set(data.index.astype(str))
    if missing:
        raise ValueError(
            f"Retained sample IDs not found in intensity data: {sorted(missing)[:5]}..."
        )
    subset = data.loc[retained_sample_ids]

    if config.input_format == "proteins_as_rows":
        out_df = subset.T.reset_index()
        if out_df.columns[0] != config.index_col:
            out_df = out_df.rename(columns={out_df.columns[0]: config.index_col})
    elif config.input_format == "proteins_as_columns":
        out_df = subset.reset_index()
        if out_df.columns[0] != config.index_col:
            out_df = out_df.rename(columns={out_df.columns[0]: config.index_col})
    else:
        raise ValueError(f"Unsupported input_format: {config.input_format}")

    out_int = tmp_dir / f"intensities_subset{suffix}"
    out_df.to_csv(out_int, sep=sep, index=False)

    subset_annotation: Optional[str] = None
    if config.sample_annotation:
        subset_annotation = str(
            _write_subset_annotation(config, retained_sample_ids, tmp_dir, sep)
        )

    return str(out_int), subset_annotation


def _write_subset_annotation(
    config: ProtriderConfig,
    retained_sample_ids: List[str],
    tmp_dir: Path,
    sep: str,
) -> Path:
    anno = read_annotation_file(config.sample_annotation)
    id_col = _detect_sample_id_column(anno, config.index_col)

    if id_col is not None:
        indexed = anno.set_index(id_col)
        missing = [
            sid for sid in retained_sample_ids if sid not in indexed.index.astype(str)
        ]
        if missing:
            raise ValueError(
                f"Retained sample IDs not found in annotation '{id_col}': {missing[:5]}..."
            )
        subset = indexed.loc[retained_sample_ids].reset_index()
    else:
        full_ids = _read_sample_ids(config)
        pos = {sid: i for i, sid in enumerate(full_ids)}
        row_indices = [pos[sid] for sid in retained_sample_ids]
        subset = anno.iloc[row_indices].reset_index(drop=True)

    src_suffix = Path(config.sample_annotation).suffix or ".tsv"
    out_path = tmp_dir / f"annotation_subset{src_suffix}"
    subset.to_csv(out_path, sep=sep, index=False)
    return out_path


def _detect_sample_id_column(
    anno: pd.DataFrame, index_col: str
) -> Optional[str]:
    for candidate in _SAMPLE_ID_COLUMN_CANDIDATES:
        if candidate in anno.columns:
            return candidate
    if index_col in anno.columns:
        return index_col
    return None


def _run_single_stability_iteration(
    config: ProtriderConfig,
    retained_sample_ids: List[str],
    iteration_seed: int,
    tmp_dir: Path,
) -> pd.DataFrame:
    """
    Run one stability iteration in an isolated directory.

    Uses a fresh checkpoint path under tmp_dir so the full-cohort model.pt is
    never loaded.
    """
    import torch

    subset_intensities, subset_annotation = _subset_input_files(
        config, retained_sample_ids, tmp_dir
    )

    iter_config = replace(
        config,
        out_dir=str(tmp_dir),
        input_intensities=subset_intensities,
        sample_annotation=subset_annotation,
        checkpoint_path=str(tmp_dir / "model.pt"),
        seed=iteration_seed,
        cohort_stability=False,
    )

    torch.manual_seed(iteration_seed)
    np.random.seed(iteration_seed)

    from protrider.pipeline import run as run_pipeline

    result, _, _, _ = run_pipeline(iter_config)
    long_df = result.to_long_df(include_all=True)
    return long_df[STABILITY_METRIC_COLUMNS].copy()


def _summarize_stability(
    iteration_summaries: List[pd.DataFrame],
    baseline_summary: pd.DataFrame,
    n_runs_requested: int,
) -> pd.DataFrame:
    """Aggregate per-iteration summaries; outer union over sample-protein keys."""
    baseline_keys = _build_baseline_table(baseline_summary)

    combined = pd.concat(iteration_summaries, ignore_index=True)
    combined["sampleID"] = combined["sampleID"].astype(str)
    combined["proteinID"] = combined["proteinID"].astype(str)

    grouped = combined.groupby(["sampleID", "proteinID"], dropna=False)

    bs_n_observed = grouped.size().rename("BS_N_OBSERVED")
    outlier_count = grouped["PROTEIN_outlier"].sum().rename(
        "PROTEIN_outlier_call_count"
    )

    agg_parts = [bs_n_observed, outlier_count]
    for col, prefix in [
        ("PROTEIN_FC", "PROTEIN_FC"),
        ("PROTEIN_LOG2FC", "PROTEIN_LOG2FC"),
        ("PROTEIN_ZSCORE", "PROTEIN_ZSCORE"),
        ("PROTEIN_PVALUE", "PROTEIN_PVALUE"),
        ("PROTEIN_PADJ", "PROTEIN_PADJ"),
    ]:
        agg_parts.append(_metric_stability_table(grouped[col], prefix))

    stability = pd.concat(agg_parts, axis=1).reset_index()

    stability["BS_N_RUNS_REQUESTED"] = n_runs_requested
    stability["BS_N_MISSING"] = (
        stability["BS_N_RUNS_REQUESTED"] - stability["BS_N_OBSERVED"]
    )
    stability["BS_OBSERVED_FRACTION"] = (
        stability["BS_N_OBSERVED"] / stability["BS_N_RUNS_REQUESTED"]
    )
    # Rate over observed iterations (main stability score for outlier calls)
    stability["PROTEIN_outlier_call_rate"] = (
        stability["PROTEIN_outlier_call_count"] / stability["BS_N_OBSERVED"]
    )
    stability.loc[stability["BS_N_OBSERVED"] == 0, "PROTEIN_outlier_call_rate"] = (
        np.nan
    )
    # Rate over all requested iterations (includes missing pairs as non-calls)
    stability["PROTEIN_outlier_call_rate_all_runs"] = (
        stability["PROTEIN_outlier_call_count"] / stability["BS_N_RUNS_REQUESTED"]
    )

    merged = stability.merge(
        baseline_keys,
        on=["sampleID", "proteinID"],
        how="outer",
    )
    merged["in_full_run"] = merged["in_full_run"].fillna(False).astype(bool)
    merged["BS_N_OBSERVED"] = merged["BS_N_OBSERVED"].fillna(0).astype(int)
    merged["BS_N_MISSING"] = merged["BS_N_RUNS_REQUESTED"] - merged["BS_N_OBSERVED"]
    merged["BS_OBSERVED_FRACTION"] = (
        merged["BS_N_OBSERVED"] / merged["BS_N_RUNS_REQUESTED"]
    )
    merged["PROTEIN_outlier_call_rate_all_runs"] = (
        merged["PROTEIN_outlier_call_count"].fillna(0)
        / merged["BS_N_RUNS_REQUESTED"]
    )
    merged.loc[
        merged["BS_N_OBSERVED"] == 0, "PROTEIN_outlier_call_rate"
    ] = np.nan

    return merged


def _build_baseline_table(baseline_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_summary.copy()
    baseline["sampleID"] = baseline["sampleID"].astype(str)
    baseline["proteinID"] = baseline["proteinID"].astype(str)

    rename = {
        "PROTEIN_outlier": "PROTEIN_outlier_full",
        "PROTEIN_FC": "PROTEIN_FC_full",
        "PROTEIN_LOG2FC": "PROTEIN_LOG2FC_full",
        "PROTEIN_ZSCORE": "PROTEIN_ZSCORE_full",
        "PROTEIN_PVALUE": "PROTEIN_PVALUE_full",
        "PROTEIN_PADJ": "PROTEIN_PADJ_full",
        "PROTEIN_LOG2INT": "PROTEIN_LOG2INT_full",
        "PROTEIN_EXPECTED_LOG2INT": "PROTEIN_EXPECTED_LOG2INT_full",
        "PROTEIN_INT": "PROTEIN_INT_full",
    }
    keep = ["sampleID", "proteinID"] + [c for c in rename if c in baseline.columns]
    out = baseline[keep].rename(columns=rename)
    out["in_full_run"] = True
    return out


def _metric_stability_table(series_group, prefix: str) -> pd.DataFrame:
    def _std(s):
        return np.nan if s.count() < 2 else s.std(ddof=1)

    def _se(s):
        n = s.count()
        return np.nan if n < 2 else s.std(ddof=1) / np.sqrt(n)

    return series_group.agg(
        **{
            f"{prefix}_mean": "mean",
            f"{prefix}_median": "median",
            f"{prefix}_sd": _std,
            f"{prefix}_se": _se,
            f"{prefix}_q025": lambda s: s.quantile(0.025),
            f"{prefix}_q975": lambda s: s.quantile(0.975),
        }
    )
