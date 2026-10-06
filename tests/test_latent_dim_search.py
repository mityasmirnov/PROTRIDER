"""
Regression test for upstream fix ab08de8 (docs/19 §5.4): the binary-search
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
import pytest

from protrider import ProtriderConfig, run


def _tiny_intensities(path, n_proteins=8, n_samples=20, seed=7):
    """Few proteins relative to samples so q (OHT) * 3 exceeds n_proteins."""
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


class TestBinarySearchLatentDimRespectsKMax:
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
        assert enc_dims.max() <= n_proteins, (
            f"binary search tested latent_dim={enc_dims.max()} > n_proteins={n_proteins}; "
            "R was not capped at k_max"
        )
