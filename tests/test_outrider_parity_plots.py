"""
Tests for OUTRIDER-parity plotting helpers (volcano, intensity rank, heatmap, CV).
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from protrider import plots


def _write_summary(out_dir, n_samples=8, n_proteins=12, seed=0):
    rng = np.random.default_rng(seed)
    samples = [f"s{i}" for i in range(n_samples)]
    proteins = [f"p{i}" for i in range(n_proteins)]
    rows = []
    for sample in samples:
        for protein in proteins:
            z = float(rng.normal())
            log2fc = float(rng.normal(scale=0.5))
            pval = float(rng.uniform(1e-6, 1.0))
            rows.append(
                {
                    "sampleID": sample,
                    "proteinID": protein,
                    "PROTEIN_LOG2INT": float(rng.normal(loc=5.0, scale=1.0)),
                    "PROTEIN_EXPECTED_LOG2INT": float(rng.normal(loc=5.0, scale=0.8)),
                    "PROTEIN_INT": float(rng.uniform(100, 1000)),
                    "PROTEIN_ZSCORE": z,
                    "PROTEIN_PVALUE": pval,
                    "PROTEIN_PADJ": min(1.0, pval * 5),
                    "PROTEIN_LOG2FC": log2fc,
                    "PROTEIN_FC": float(2 ** log2fc),
                    "PROTEIN_outlier": abs(z) > 2.0,
                }
            )
    # Guarantee at least one outlier for default sample/protein selection.
    rows[0]["PROTEIN_outlier"] = True
    rows[0]["PROTEIN_ZSCORE"] = 3.5
    rows[0]["PROTEIN_PVALUE"] = 1e-5
    rows[0]["PROTEIN_PADJ"] = 1e-4
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "protrider_summary.csv", index=False)
    return summary, samples, proteins


def _write_matrices(out_dir, samples, proteins, seed=1):
    rng = np.random.default_rng(seed)
    n_s, n_p = len(samples), len(proteins)
    raw = pd.DataFrame(
        rng.normal(loc=5.0, scale=1.0, size=(n_p, n_s)),
        index=proteins,
        columns=samples,
    )
    raw.index.name = "proteinID"
    raw.to_csv(out_dir / "processed_input.csv")

    expected = raw + rng.normal(scale=0.2, size=raw.shape)
    expected.index.name = "proteinID"
    expected.to_csv(out_dir / "output.csv")

    resid = raw - expected
    resid.index.name = "proteinID"
    resid.to_csv(out_dir / "residuals.csv")
    return raw, expected, resid


@pytest.fixture
def parity_outdir(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _, samples, proteins = _write_summary(out_dir)
    _write_matrices(out_dir, samples, proteins)
    return out_dir


def test_plot_volcano_creates_png(parity_outdir):
    result = plots.plot_volcano(str(parity_outdir), sample_id="s0")
    png = parity_outdir / "plots" / "volcano.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert result is not None


def test_plot_volcano_zscore_metric(parity_outdir):
    result = plots.plot_volcano(
        str(parity_outdir), sample_id="s0", y_metric="zscore"
    )
    assert (parity_outdir / "plots" / "volcano.png").exists()
    assert result is not None


def test_plot_intensity_rank_creates_png(parity_outdir):
    result = plots.plot_intensity_rank(str(parity_outdir), protein_id="p0")
    png = parity_outdir / "plots" / "intensity_rank.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert result is not None


def test_plot_gene_sample_heatmap_creates_pngs(parity_outdir):
    raw_plot, norm_plot = plots.plot_gene_sample_heatmap(
        str(parity_outdir), n_proteins=6
    )
    raw_png = parity_outdir / "plots" / "gene_sample_heatmap_raw.png"
    norm_png = parity_outdir / "plots" / "gene_sample_heatmap_normalized.png"
    assert raw_png.exists() and raw_png.stat().st_size > 0
    assert norm_png.exists() and norm_png.stat().st_size > 0
    assert raw_plot is not None
    assert norm_plot is not None


def test_plot_cv_before_after_creates_png(parity_outdir):
    result = plots.plot_cv_before_after(str(parity_outdir))
    png = parity_outdir / "plots" / "cv_before_after.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert result is not None


def test_parity_plots_missing_summary_return_none(tmp_path):
    out_dir = tmp_path / "empty"
    out_dir.mkdir()
    assert plots.plot_volcano(str(out_dir)) is None
    assert plots.plot_intensity_rank(str(out_dir)) is None


def test_parity_plots_accept_in_memory_frames(tmp_path):
    out_dir = tmp_path / "mem"
    out_dir.mkdir()
    summary, samples, proteins = _write_summary(out_dir)
    raw, expected, resid = _write_matrices(out_dir, samples, proteins)

    assert plots.plot_volcano(
        str(out_dir), sample_id="s0", protrider_summary=summary
    ) is not None
    assert plots.plot_intensity_rank(
        str(out_dir), protein_id="p0", protrider_summary=summary
    ) is not None
    raw_plot, norm_plot = plots.plot_gene_sample_heatmap(
        str(out_dir),
        n_proteins=5,
        processed_input=raw.reset_index(),
        output_data=expected.reset_index(),
    )
    assert raw_plot is not None and norm_plot is not None
    assert plots.plot_cv_before_after(
        str(out_dir), processed_input=raw, residuals=resid
    ) is not None
