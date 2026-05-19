"""
Tests for co-outlier patient stratification (directional Jaccard on Z-scores).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from protrider import CoOutlierSimilarity, compute_cooutlier_similarity, run
from protrider.cooutlier_similarity import LOW_BURDEN_LABEL

from tests.test_latent_space import _fast_latent_config, _make_synthetic_intensities


def _directional_zscores() -> pd.DataFrame:
    """Samples × proteins with two up/down cohorts and one zero-burden sample."""
    proteins = ["pA", "pB", "pC", "pD", "pE", "pF", "pBG", "pConflict"]
    samples = ["s1", "s2", "s3", "s4", "s5", "s6", "s_zero"]
    z = np.zeros((len(samples), len(proteins)))
    # Up cohort shares pA–pC
    for i in range(3):
        z[i, 0:3] = 4.0
    # Down cohort shares pD–pF
    for i in range(3, 6):
        z[i, 3:6] = -4.0
    # Background
    z[:, 6] = 0.1
    # Opposite directions on same protein (should not match)
    z[0, 7] = 4.0
    z[3, 7] = -4.0
    return pd.DataFrame(z, index=samples, columns=proteins)


class TestComputeCooutlierSimilarity:
    def test_directional_jaccard_patterns(self):
        zscores = _directional_zscores()
        result = compute_cooutlier_similarity(zscores, z_threshold=3.0, min_anomalies=1)

        assert isinstance(result, CoOutlierSimilarity)
        sim = result.similarity
        assert sim.shape == (7, 7)
        assert np.allclose(sim.values, sim.values.T, atol=1e-10)
        assert sim.min().min() >= 0.0
        assert sim.max().max() <= 1.0 + 1e-10

        for sid in ["s1", "s2", "s3", "s4", "s5", "s6"]:
            assert sim.loc[sid, sid] == pytest.approx(1.0)

        assert sim.loc["s_zero", "s_zero"] == pytest.approx(0.0)

        within_up = sim.loc["s1", ["s2", "s3"]].mean()
        within_down = sim.loc["s4", ["s5", "s6"]].mean()
        between = sim.loc["s1", ["s4", "s5", "s6"]].mean()
        assert within_up > between
        assert within_down > between

        # s1/s4 share pConflict in opposite directions → no match on that protein
        assert sim.loc["s1", "s4"] < within_up

        burden = result.burden
        assert set(burden.columns) >= {"sampleID", "n_up", "n_down", "n_total"}
        assert burden.loc[burden["sampleID"] == "s1", "n_up"].iloc[0] == 4
        assert burden.loc[burden["sampleID"] == "s4", "n_down"].iloc[0] == 4
        assert burden.loc[burden["sampleID"] == "s_zero", "n_total"].iloc[0] == 0

        subp = result.subpopulations
        assert len(subp) == 7
        zero_row = subp.loc[subp["sampleID"] == "s_zero"].iloc[0]
        assert zero_row["subpopulation"] == LOW_BURDEN_LABEL
        assert not bool(zero_row["eligible_for_clustering"])

        for col in ["PC1", "PC2"]:
            assert col in result.pca_coordinates.columns
        for col in ["UMAP1", "UMAP2"]:
            assert col in result.umap_coordinates.columns
        for col in ["TSNE1", "TSNE2"]:
            assert col in result.tsne_coordinates.columns

        info = result.info.iloc[0]
        assert info["similarity_method"] == "directional_jaccard_same_direction"
        assert info["z_threshold"] == 3.0

        eligible = subp[subp["eligible_for_clustering"].astype(bool)]
        assert not eligible["subpopulation"].isna().any()
        assert not (eligible["subpopulation"].astype(str).str.len() == 0).any()
        assert eligible["subpopulation"].str.startswith("subpopulation_").all()

    def test_single_sample_returns_none(self):
        zscores = pd.DataFrame([[1.0]], index=["only"], columns=["p1"])
        assert compute_cooutlier_similarity(zscores) is None

    def test_save_writes_expected_files(self, tmp_path):
        result = compute_cooutlier_similarity(_directional_zscores())
        out_dir = tmp_path / "out"
        written = result.save(str(out_dir))

        assert (out_dir / "cooutlier_patient_similarity.csv").exists()
        assert (out_dir / "cooutlier_patient_subpopulations.csv").exists()
        assert (out_dir / "cooutlier_patient_burden.csv").exists()
        assert (out_dir / "cooutlier_patient_info.csv").exists()
        assert "cooutlier_patient_similarity" in written


class TestCooutlierPipeline:
    def test_run_and_wide_save_export_cooutlier_files(self, tmp_path):
        intensities_path = tmp_path / "intensities.csv"
        _make_synthetic_intensities(intensities_path)
        config = _fast_latent_config(
            tmp_path,
            intensities_path,
            z_threshold=1.5,
            cooutlier_min_anomalies=1,
        )

        result, model_info, *_ = run(config)
        assert result.cooutlier_similarity is not None

        model_info.save(config.out_dir)
        result.save(config.out_dir, format="wide")

        out = Path(config.out_dir)
        assert (out / "cooutlier_patient_similarity.csv").exists()
        assert (out / "cooutlier_patient_subpopulations.csv").exists()
        assert (out / "cooutlier_patient_burden.csv").exists()
        assert (out / "cooutlier_patient_info.csv").exists()


class TestCooutlierPublicApi:
    def test_import_from_protrider(self):
        from protrider import CoOutlierSimilarity, compute_cooutlier_similarity

        assert CoOutlierSimilarity is not None
        assert compute_cooutlier_similarity is not None
