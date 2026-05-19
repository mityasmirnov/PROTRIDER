"""
Tests for patient similarity and subpopulation export.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from protrider import ProtriderConfig, run
from protrider.patient_similarity import PatientSimilarity, compute_patient_similarity
from protrider.pipeline import Result

from tests.test_latent_space import _fast_latent_config, _make_synthetic_intensities


def _two_cluster_latent_samples(n_per_cluster: int = 6, n_latent: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(n_per_cluster, n_latent)) + 5.0
    cluster_b = rng.normal(loc=0.0, scale=0.1, size=(n_per_cluster, n_latent)) - 5.0
    z = np.vstack([cluster_a, cluster_b])
    index = [f"sample_{i}" for i in range(z.shape[0])]
    columns = [f"latent_{j + 1}" for j in range(n_latent)]
    return pd.DataFrame(z, index=index, columns=columns)


class TestComputePatientSimilarity:
    def test_two_cluster_similarity_and_subpopulations(self):
        latent_samples = _two_cluster_latent_samples()
        result = compute_patient_similarity(latent_samples)

        assert isinstance(result, PatientSimilarity)
        sim = result.similarity
        assert sim.shape[0] == sim.shape[1] == latent_samples.shape[0]
        assert np.allclose(sim.values, sim.values.T, atol=1e-10)
        assert np.allclose(np.diag(sim.values), 1.0)
        assert sim.min().min() >= 0.0
        assert sim.max().max() <= 1.0 + 1e-10

        within_a = sim.loc["sample_0", ["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]]
        between = sim.loc["sample_0", ["sample_6", "sample_7", "sample_8", "sample_9", "sample_10", "sample_11"]]
        assert within_a.mean() > between.mean()

        subp = result.subpopulations
        assert subp is not None
        for col in [
            "sampleID",
            "subpopulation",
            "silhouette_score_for_selected_k",
            "selected_k",
            "clustering_method",
            "similarity_method",
        ]:
            assert col in subp.columns
        assert len(subp) == latent_samples.shape[0]

        pca = result.pca_coordinates
        assert pca is not None
        assert {"sampleID", "PC1", "PC2", "subpopulation"}.issubset(pca.columns)

        umap = result.umap_coordinates
        assert umap is not None
        assert {"sampleID", "UMAP1", "UMAP2", "subpopulation"}.issubset(umap.columns)

        tsne = result.tsne_coordinates
        assert tsne is not None
        assert {"sampleID", "TSNE1", "TSNE2", "subpopulation"}.issubset(tsne.columns)

        info = result.info
        assert info is not None
        assert info.loc[0, "status"] in {
            "ok",
            "degenerate_all_samples_identical",
            "not_enough_samples_for_clustering",
            "clustering_degenerate_single_cluster",
            "clustering_failed",
        }

    def test_single_sample_returns_none(self):
        latent_samples = pd.DataFrame(
            [[1.0, 2.0]],
            index=["only_sample"],
            columns=["latent_1", "latent_2"],
        )
        assert compute_patient_similarity(latent_samples) is None

    def test_identical_samples_identity_matrix(self):
        latent_samples = pd.DataFrame(
            np.ones((4, 3)),
            index=[f"s{i}" for i in range(4)],
            columns=["latent_1", "latent_2", "latent_3"],
        )
        result = compute_patient_similarity(latent_samples, min_samples_for_clustering=4)
        assert result is not None
        assert np.allclose(result.similarity.values, np.eye(4))
        assert result.info.loc[0, "status"] == "degenerate_all_samples_identical"
        assert result.umap_coordinates is None
        assert result.tsne_coordinates is None
        assert result.info.loc[0, "umap_status"] == "skipped_degenerate_latents"
        assert result.info.loc[0, "tsne_status"] == "skipped_degenerate_latents"

    def test_save_writes_expected_files(self, tmp_path):
        latent_samples = _two_cluster_latent_samples()
        result = compute_patient_similarity(latent_samples)
        out_dir = tmp_path / "out"
        written = result.save(str(out_dir))

        assert (out_dir / "patient_similarity.csv").exists()
        assert (out_dir / "patient_subpopulations.csv").exists()
        assert (out_dir / "patient_latent_pca.csv").exists()
        assert (out_dir / "patient_latent_umap.csv").exists()
        assert (out_dir / "patient_latent_tsne.csv").exists()
        assert (out_dir / "patient_similarity_info.csv").exists()
        assert "patient_similarity" in written
        assert "patient_latent_umap" in written
        assert "patient_latent_tsne" in written


class TestPatientSimilarityPipeline:
    def test_run_and_wide_save_export_patient_files(self, tmp_path):
        intensities_path = tmp_path / "intensities.csv"
        _make_synthetic_intensities(intensities_path)
        config = _fast_latent_config(tmp_path, intensities_path)

        result, model_info, *_ = run(config)
        assert result.latent_space is not None
        assert result.patient_similarity is not None

        model_info.save(config.out_dir)
        result.save(config.out_dir, format="wide")

        out = Path(config.out_dir)
        assert (out / "patient_similarity.csv").exists()
        assert (out / "patient_subpopulations.csv").exists()
        assert (out / "patient_latent_pca.csv").exists()
        assert (out / "patient_latent_umap.csv").exists()
        assert (out / "patient_latent_tsne.csv").exists()
        assert (out / "patient_similarity_info.csv").exists()
        assert (out / "latent_samples.csv").exists()

    def test_missing_latent_skips_patient_similarity(self):
        result = Result(
            dataset=None,
            df_out=pd.DataFrame(),
            df_res=pd.DataFrame(),
            df_pvals=pd.DataFrame(),
            df_pvals_one_sided=pd.DataFrame(),
            df_presence=None,
            df_Z=pd.DataFrame(),
            df_pvals_adj=pd.DataFrame(),
            log2fc=np.array([]),
            fc=np.array([]),
            n_out_median=0,
            n_out_max=0,
            n_out_total=0,
            latent_space=None,
            patient_similarity=None,
        )
        assert result.patient_similarity is None
