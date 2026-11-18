"""Normalised intensity, robust (Student-t) p-values and confidence-interval merge."""

import tempfile

import numpy as np
import pandas as pd

from protrider import ProtriderConfig, run
from protrider.stability import merge_confidence_intervals


def _run(tmp_dir, protein_intensities_path, covariates_path, index_col, **kw):
    config = ProtriderConfig(out_dir=tmp_dir, input_intensities=str(protein_intensities_path),
                             sample_annotation=str(covariates_path), index_col=index_col,
                             cov_used=['AGE', 'SEX'], n_epochs=2, gs_epochs=2, find_q_method='5',
                             pval_dist='gaussian', verbose=False, **kw)
    result, *_ = run(config)
    return result


def test_normalised_intensity(protein_intensities_path, covariates_path, protein_intensities_index_col):
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(tmp, protein_intensities_path, covariates_path, protein_intensities_index_col)
        expect = r.dataset.data - r.df_out + r.dataset.data.mean(axis=0)
        pd.testing.assert_frame_equal(r.df_norm, expect)
        # same missing cells as the observed data; protein means are kept
        assert (r.df_norm.isna() == r.dataset.data.isna()).all().all()
        np.testing.assert_allclose(r.df_norm.mean(axis=0) - r.dataset.data.mean(axis=0),
                                   (r.dataset.data - r.df_out).mean(axis=0), atol=1e-8)
        long = r.to_long_df(include_all=True)
        assert {'PROTEIN_NORM_LOG2INT', 'PROTEIN_NORM_INT'} <= set(long.columns)
        assert 'PROTEIN_PVALUE_ROBUST' not in long.columns  # flag off


def test_robust_pvals(protein_intensities_path, covariates_path, protein_intensities_index_col):
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(tmp, protein_intensities_path, covariates_path, protein_intensities_index_col, robust_pvals=True)
        long = r.to_long_df(include_all=True)
        assert {'PROTEIN_PVALUE_ROBUST', 'PROTEIN_PADJ_ROBUST'} <= set(long.columns)
        ok = long.PROTEIN_PVALUE_ROBUST.notna()
        assert ok.any()
        assert ((long.PROTEIN_PVALUE_ROBUST[ok] >= 0) & (long.PROTEIN_PVALUE_ROBUST[ok] <= 1)).all()
        assert (long.PROTEIN_PADJ_ROBUST[ok] >= long.PROTEIN_PVALUE_ROBUST[ok] - 1e-12).all()
        # primary (gaussian) p-values are unchanged by the flag
        r0 = _run(tmp + "_0", protein_intensities_path, covariates_path, protein_intensities_index_col, seed=1)
        assert r0.df_pvals_robust is None


def test_merge_confidence_intervals():
    long = pd.DataFrame({"sampleID": ["s1", "s1", "s2"], "proteinID": ["p1", "p2", "p1"], "PROTEIN_ZSCORE": [-4.0, 0.1, 1.0]})
    bs = pd.DataFrame({"sampleID": ["s1", "s2"], "proteinID": ["p1", "p1"],
                       "PROTEIN_ZSCORE_median": [-3.9, 0.9], "PROTEIN_ZSCORE_q025": [-4.8, 0.2],
                       "PROTEIN_ZSCORE_q975": [-3.0, 1.6], "PROTEIN_ZSCORE_sd": [0.4, 0.3],
                       "PROTEIN_outlier_call_rate": [0.9, 0.0], "BS_N_RUNS_COMPLETED": [100, 100]})
    m = merge_confidence_intervals(long, bs)
    assert len(m) == 3 and "PROTEIN_ZSCORE_sd" not in m.columns
    assert m.loc[(m.sampleID == "s1") & (m.proteinID == "p1"), "PROTEIN_ZSCORE_q025"].item() == -4.8
    assert m.loc[(m.sampleID == "s1") & (m.proteinID == "p2"), "PROTEIN_ZSCORE_q975"].isna().all()
