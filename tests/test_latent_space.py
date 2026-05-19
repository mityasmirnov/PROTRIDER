"""
Tests for latent-space extraction and export.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from protrider import ProtriderConfig, run
from protrider.model import ProtriderAutoencoder
from protrider.pipeline import Result


def _make_synthetic_intensities(path: Path, n_proteins: int = 30, n_samples: int = 24) -> list[str]:
    """Synthetic matrix large enough for OHT latent-dimension selection."""
    rng = np.random.default_rng(42)
    proteins = [f"PROT_{i:03d}" for i in range(n_proteins)]
    samples = [f"sample_{i}" for i in range(n_samples)]
    # low-rank signal + noise so OHT selects a nontrivial q
    latent = rng.normal(size=(n_samples, 5))
    loadings = rng.normal(size=(5, n_proteins))
    signal = latent @ loadings
    noise = rng.normal(scale=0.5, size=(n_samples, n_proteins))
    values = np.exp(signal + noise) * 1000.0
    values = np.clip(values, 100.0, None)
    df = pd.DataFrame(values.T, index=proteins, columns=samples)
    df.index.name = "protein_ID"
    df = df.reset_index()
    df.to_csv(path, index=False)
    return samples


def _make_synthetic_annotation(path: Path, samples: list[str], batch_values=None):
    if batch_values is None:
        batch_values = ["A", "B"] * (len(samples) // 2 + 1)
    df = pd.DataFrame(
        {
            "sample_ID": samples,
            "batch": [batch_values[i % len(batch_values)] for i in range(len(samples))],
        }
    )
    df.to_csv(path, sep="\t", index=False)


def _fast_latent_config(tmp_path, intensities_path, **overrides) -> ProtriderConfig:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        out_dir=str(out_dir),
        input_intensities=str(intensities_path),
        index_col="protein_ID",
        find_q_method="OHT",
        n_layers=1,
        init_pca=True,
        autoencoder_training=False,
        n_epochs=1,
        device="cpu",
        common_degrees_freedom=False,
        n_jobs=1,
        verbose=False,
    )
    defaults.update(overrides)
    return ProtriderConfig(**defaults)


class TestLatentSpaceExport:
    def test_oht_linear_exports_all_latent_files(self, tmp_path):
        intensities_path = tmp_path / "intensities.csv"
        samples = _make_synthetic_intensities(intensities_path)
        config = _fast_latent_config(tmp_path, intensities_path)

        result, model_info, *_ = run(config)
        assert isinstance(result, Result)
        assert result.latent_space is not None

        model_info.save(config.out_dir)
        result.save(config.out_dir, format="wide")

        out_dir = Path(config.out_dir)
        assert (out_dir / "latent_samples.csv").exists()
        assert (out_dir / "latent_protein_loadings_svd.csv").exists()
        assert (out_dir / "latent_protein_loadings_decoder.csv").exists()
        assert (out_dir / "residuals.csv").exists()
        assert (out_dir / "pvals.csv").exists()
        assert (out_dir / "zscores.csv").exists()
        assert (out_dir / "additional_info.csv").exists()

        q = int(model_info.q.item() if hasattr(model_info.q, "item") else model_info.q)
        n_samples = len(samples)
        n_proteins = result.dataset.data.shape[1]

        z = pd.read_csv(out_dir / "latent_samples.csv", index_col=0)
        svd_loadings = pd.read_csv(out_dir / "latent_protein_loadings_svd.csv", index_col=0)
        dec_loadings = pd.read_csv(out_dir / "latent_protein_loadings_decoder.csv", index_col=0)

        assert z.shape == (n_samples, q)
        assert svd_loadings.shape == (n_proteins, q)
        assert dec_loadings.shape == (n_proteins, q)
        assert list(z.index) == list(result.dataset.data.index)
        assert list(svd_loadings.index) == list(result.dataset.data.columns)
        assert list(dec_loadings.index) == list(result.dataset.data.columns)

    def test_multilayer_skips_decoder_loadings(self, tmp_path):
        intensities_path = tmp_path / "intensities.csv"
        _make_synthetic_intensities(intensities_path)
        config = _fast_latent_config(
            tmp_path,
            intensities_path,
            n_layers=2,
            h_dim=4,
        )

        result, model_info, *_ = run(config)
        result.save(config.out_dir, format="wide")

        out_dir = Path(config.out_dir)
        assert (out_dir / "latent_samples.csv").exists()
        assert (out_dir / "latent_protein_loadings_svd.csv").exists()
        assert not (out_dir / "latent_protein_loadings_decoder.csv").exists()
        assert result.latent_space.protein_loadings_decoder is None

    def test_oht_with_covariates_exports_latent_samples(self, tmp_path):
        intensities_path = tmp_path / "intensities.csv"
        samples = _make_synthetic_intensities(intensities_path)
        annotation_path = tmp_path / "annotations.tsv"
        _make_synthetic_annotation(annotation_path, samples)

        config = _fast_latent_config(
            tmp_path,
            intensities_path,
            sample_annotation=str(annotation_path),
            cov_used=["batch"],
        )

        result, model_info, *_ = run(config)
        result.save(config.out_dir, format="wide")

        out_dir = Path(config.out_dir)
        assert (out_dir / "latent_samples.csv").exists()
        assert (out_dir / "latent_protein_loadings_svd.csv").exists()
        assert (out_dir / "latent_protein_loadings_decoder.csv").exists()

        q = int(model_info.q.item() if hasattr(model_info.q, "item") else model_info.q)
        dec_loadings = pd.read_csv(out_dir / "latent_protein_loadings_decoder.csv", index_col=0)
        svd_loadings = pd.read_csv(out_dir / "latent_protein_loadings_svd.csv", index_col=0)
        n_proteins = result.dataset.data.shape[1]

        assert dec_loadings.shape == (n_proteins, q)
        assert svd_loadings.shape == (n_proteins, q)

    def test_existing_statistical_outputs_still_written(self, tmp_path):
        intensities_path = tmp_path / "intensities.csv"
        _make_synthetic_intensities(intensities_path)
        config = _fast_latent_config(tmp_path, intensities_path)

        result, model_info, *_ = run(config)
        assert result.latent_space is not None
        result.save(config.out_dir, format="wide")

        out_dir = Path(config.out_dir)
        for name in (
            "processed_input.csv",
            "output.csv",
            "residuals.csv",
            "pvals.csv",
            "pvals_adj.csv",
            "zscores.csv",
            "log2fc.csv",
            "fc.csv",
            "latent_samples.csv",
        ):
            assert (out_dir / name).exists(), f"missing {name}"


class TestInitializeWPCA:
    def test_covariate_columns_preserved_at_trailing_positions(self):
        n_samples, n_proteins, n_cov, q = 8, 12, 2, 3
        rng = np.random.default_rng(0)

        model = ProtriderAutoencoder(
            in_dim=n_proteins,
            latent_dim=q,
            n_layers=1,
            n_cov=n_cov,
            prot_means=None,
        )
        model.double()

        enc_layer = model.encoder.model
        dec_layer = model.decoder.model

        enc_layer.weight.data = torch.randn(q, n_proteins + n_cov, dtype=torch.float64)
        dec_layer.weight.data = torch.randn(n_proteins, q + n_cov, dtype=torch.float64)

        cov_enc_before = enc_layer.weight.data[:, n_proteins:].clone()
        cov_dec_before = dec_layer.weight.data[:, q:].clone()

        centered = rng.normal(size=(n_samples, n_proteins))
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        vt_q = vt[:q]
        prot_means = np.nanmean(centered, axis=0, keepdims=True)

        model.initialize_wPCA(vt_q, prot_means, n_cov=n_cov)

        assert torch.allclose(enc_layer.weight.data[:, n_proteins:], cov_enc_before)
        assert torch.allclose(dec_layer.weight.data[:, q:], cov_dec_before)
        assert torch.allclose(
            enc_layer.weight.data[:, :n_proteins],
            torch.from_numpy(vt_q).to(dtype=torch.float64),
        )
        assert torch.allclose(
            dec_layer.weight.data[:, :q],
            torch.from_numpy(vt_q.T).to(dtype=torch.float64),
        )
