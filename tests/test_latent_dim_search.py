"""
Regression tests for upstream fix ab08de8 (docs/19 §5.4): the binary-search
latent-dimension method's right bound must never exceed the number of
proteins (k_max). Before the fix, `k_max` was computed but unused, and
`R = int(q * 3)` could request `train_and_eval_q(latent_dim)` with
`latent_dim > n_proteins`, which silently truncates `dataset.Vt[:latent_dim]`
to fewer columns than requested instead of raising — a quiet quality bug,
not a crash, which is why no prior test caught it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from protrider import ProtriderConfig, run
from protrider.model.model_helper import _binary_search_bounds


class TestBinarySearchBounds:
    """Direct, deterministic test of the fixed arithmetic — no training."""

    def test_r_capped_at_k_max_when_q_times_3_would_exceed_it(self):
        L, M, R = _binary_search_bounds(q=5, factor=2, k_max=10)
        assert (L, M, R) == (2, 5, 10)  # q*3=15 > k_max=10 -> capped

    def test_r_unchanged_when_q_times_3_is_within_k_max(self):
        L, M, R = _binary_search_bounds(q=2, factor=2, k_max=100)
        assert (L, M, R) == (1, 2, 6)  # q*3=6 <= k_max -> untouched

    def test_r_never_exceeds_k_max_across_a_range_of_q_and_k_max(self):
        for q in range(1, 20):
            for k_max in range(1, 20):
                _L, _M, R = _binary_search_bounds(q, factor=2, k_max=k_max)
                assert R <= k_max

    def test_l_and_m_unaffected_by_the_fix(self):
        for q in (1, 3, 7, 12):
            L, M, _R = _binary_search_bounds(q, factor=2, k_max=1000)
            assert L == max(1, q // 2)
            assert M == q


def _tiny_intensities(path, n_proteins=8, n_samples=20, seed=7):
    rng = np.random.default_rng(seed)
    proteins = [f"PROT_{i:03d}" for i in range(n_proteins)]
    samples = [f"sample_{i}" for i in range(n_samples)]
    latent = rng.normal(size=(n_samples, 3))
    loadings = rng.normal(size=(3, n_proteins))
    signal = latent @ loadings
    noise = rng.normal(scale=0.4, size=(n_samples, n_proteins))
    values = np.exp(signal + noise) * 1000.0
    values = np.clip(values, 100.0, None)
    df = pd.DataFrame(values.T, index=proteins, columns=samples)
    df.index.name = "protein_ID"
    df.reset_index().to_csv(path, index=False)
    return samples, n_proteins


class TestBinarySearchLatentDimEndToEnd:
    """Looser end-to-end smoke: the full bs pipeline still runs and respects k_max.

    OHT's chosen q is data-dependent, so this does not reliably exercise the
    capped branch (see TestBinarySearchBounds for that) — it only confirms
    the fix didn't break the method end-to-end.
    """

    def test_tested_dims_never_exceed_protein_count(self, tmp_path):
        intensities_path = tmp_path / "intensities.csv"
        _samples, n_proteins = _tiny_intensities(intensities_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        config = ProtriderConfig(
            out_dir=str(out_dir),
            input_intensities=str(intensities_path),
            index_col="protein_ID",
            find_q_method="bs",
            n_layers=1,
            init_pca=True,
            autoencoder_training=False,
            n_epochs=1,
            gs_epochs=1,
            patience=1,
            device="cpu",
            common_degrees_freedom=False,
            n_jobs=1,
            verbose=False,
        )

        result, _model_info, _fit_params, gs_result = run(config)
        assert result is not None
        assert gs_result is not None

        enc_dims = gs_result.enc_dim
        assert enc_dims.size > 0
        assert enc_dims.max() <= n_proteins
