"""Tests for cohort stability (subsampling) analysis."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from protrider import ProtriderConfig, run
from protrider.pipeline import Result
from protrider.stability import (
    STABILITY_METRIC_COLUMNS,
    _generate_subsample_plan,
    _run_single_stability_iteration,
    _summarize_stability,
    run_cohort_stability,
)


def _make_stability_intensities(
    path: Path, n_proteins: int = 25, n_samples: int = 35
) -> list[str]:
    rng = np.random.default_rng(0)
    proteins = [f"PROT_{i:03d}" for i in range(n_proteins)]
    samples = [f"sample_{i}" for i in range(n_samples)]
    latent = rng.normal(size=(n_samples, 5))
    loadings = rng.normal(size=(5, n_proteins))
    values = np.exp(latent @ loadings + rng.normal(scale=0.3, size=(n_samples, n_proteins)))
    values = np.clip(values, 50.0, None) * 1000.0
    df = pd.DataFrame(values.T, index=proteins, columns=samples)
    df.index.name = "protein_ID"
    df = df.reset_index()
    df.to_csv(path, sep="\t", index=False)
    return samples


def _fast_stability_config(tmp_path, intensities_path, **overrides) -> ProtriderConfig:
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
        report_all=True,
    )
    defaults.update(overrides)
    return ProtriderConfig(**defaults)


class TestStabilityConfigDefaults:
    def test_stability_fields_default_off(self):
        config = ProtriderConfig(out_dir="output", input_intensities="data.csv")
        assert config.cohort_stability is False
        assert config.cohort_stability_n_runs == 100
        assert config.cohort_stability_min_runs == 30
        assert config.cohort_stability_max_runtime_min is None
        assert config.cohort_stability_drop_fraction == 0.1
        assert config.cohort_stability_min_samples == 30
        assert config.cohort_stability_seed is None
        assert config.cohort_stability_require_oht is True
        assert config.cohort_stability_save_iteration_files is False

    def test_old_minimal_yaml_without_stability_keys(self, tmp_path):
        cfg = {
            "out_dir": str(tmp_path / "out"),
            "input_intensities": "data.csv",
        }
        import yaml

        path = tmp_path / "minimal.yaml"
        path.write_text(yaml.dump(cfg))
        from protrider import load_config

        config = load_config(path)
        assert config.cohort_stability is False


class TestStabilityConfigValidation:
    def test_invalid_n_runs(self):
        with pytest.raises(ValueError, match="cohort_stability_n_runs"):
            ProtriderConfig(
                out_dir="o",
                input_intensities="d.csv",
                cohort_stability_n_runs=0,
            )

    def test_min_runs_exceeds_n_runs(self):
        with pytest.raises(ValueError, match="cohort_stability_min_runs"):
            ProtriderConfig(
                out_dir="o",
                input_intensities="d.csv",
                cohort_stability_n_runs=5,
                cohort_stability_min_runs=10,
            )

    def test_invalid_drop_fraction(self):
        with pytest.raises(ValueError, match="cohort_stability_drop_fraction"):
            ProtriderConfig(
                out_dir="o",
                input_intensities="d.csv",
                cohort_stability_drop_fraction=1.0,
            )

    def test_invalid_min_samples(self):
        with pytest.raises(ValueError, match="cohort_stability_min_samples"):
            ProtriderConfig(
                out_dir="o",
                input_intensities="d.csv",
                cohort_stability_min_samples=1,
            )

    def test_non_oht_with_stability_require_oht(self):
        with pytest.raises(ValueError, match="cohort_stability requires find_q_method"):
            ProtriderConfig(
                out_dir="o",
                input_intensities="d.csv",
                cohort_stability=True,
                find_q_method="5",
                cohort_stability_require_oht=True,
            )


class TestResultToLongDf:
    def test_to_long_df_include_all_columns(self, tmp_path):
        intensities_path = tmp_path / "intensities.tsv"
        _make_stability_intensities(intensities_path, n_samples=24)
        config = _fast_stability_config(tmp_path, intensities_path)
        result, _, _, _ = run(config)
        long_df = result.to_long_df(include_all=True)
        assert {"sampleID", "proteinID", "PROTEIN_PADJ", "PROTEIN_outlier"}.issubset(
            long_df.columns
        )
        assert len(long_df) == 24 * 25

    def test_save_long_still_writes_summary(self, tmp_path):
        intensities_path = tmp_path / "intensities.tsv"
        _make_stability_intensities(intensities_path, n_samples=24)
        config = _fast_stability_config(tmp_path, intensities_path)
        result, _, _, _ = run(config)
        result.save(config.out_dir, format="long", include_all=True)
        summary_path = Path(config.out_dir) / "protrider_summary.csv"
        assert summary_path.exists()
        saved = pd.read_csv(summary_path)
        assert "PROTEIN_outlier" in saved.columns


class TestSummarizeStability:
    def test_mismatched_proteins_outer_union(self):
        baseline = pd.DataFrame(
            {
                "sampleID": ["s1", "s1"],
                "proteinID": ["p1", "p2"],
                "PROTEIN_outlier": [True, False],
                "PROTEIN_FC": [0.5, 1.0],
                "PROTEIN_LOG2FC": [-1.0, 0.0],
                "PROTEIN_ZSCORE": [2.0, 0.1],
                "PROTEIN_PVALUE": [0.01, 0.5],
                "PROTEIN_PADJ": [0.05, 0.5],
            }
        )
        iter0 = pd.DataFrame(
            {
                "sampleID": ["s1"],
                "proteinID": ["p1"],
                "PROTEIN_outlier": [True],
                "PROTEIN_FC": [0.4],
                "PROTEIN_LOG2FC": [-1.2],
                "PROTEIN_ZSCORE": [2.5],
                "PROTEIN_PVALUE": [0.02],
                "PROTEIN_PADJ": [0.04],
            }
        )
        iter1 = pd.DataFrame(
            {
                "sampleID": ["s1"],
                "proteinID": ["p3"],
                "PROTEIN_outlier": [False],
                "PROTEIN_FC": [1.1],
                "PROTEIN_LOG2FC": [0.2],
                "PROTEIN_ZSCORE": [0.5],
                "PROTEIN_PVALUE": [0.3],
                "PROTEIN_PADJ": [0.4],
            }
        )
        out = _summarize_stability([iter0, iter1], baseline, n_runs_requested=2)
        keys = set(zip(out["sampleID"], out["proteinID"]))
        assert keys == {("s1", "p1"), ("s1", "p2"), ("s1", "p3")}
        row_p1 = out[(out["sampleID"] == "s1") & (out["proteinID"] == "p1")].iloc[0]
        assert row_p1["BS_N_RUNS_REQUESTED"] == 2
        assert row_p1["BS_N_OBSERVED"] == 1
        assert row_p1["BS_N_MISSING"] == 1
        assert row_p1["BS_OBSERVED_FRACTION"] == 0.5
        assert row_p1["PROTEIN_outlier_call_count"] == 1
        assert row_p1["PROTEIN_outlier_call_rate"] == 1.0
        assert row_p1["PROTEIN_outlier_call_rate_all_runs"] == 0.5
        assert row_p1["PROTEIN_FC_median"] == pytest.approx(0.4)
        row_p3 = out[(out["sampleID"] == "s1") & (out["proteinID"] == "p3")].iloc[0]
        assert row_p3["BS_N_OBSERVED"] == 1
        assert not row_p3["in_full_run"]


class TestSkipSmallCohort:
    def test_skips_when_thirty_or_fewer_samples(self, tmp_path, caplog):
        intensities_path = tmp_path / "intensities.tsv"
        _make_stability_intensities(intensities_path, n_samples=30)
        config = _fast_stability_config(
            tmp_path,
            intensities_path,
            cohort_stability=True,
            cohort_stability_n_runs=2,
            cohort_stability_min_runs=1,
        )
        result, _, _, _ = run(config)
        result.save(config.out_dir, format="long", include_all=True)
        baseline = result.to_long_df(include_all=True)
        with caplog.at_level("WARNING"):
            bs = run_cohort_stability(config, baseline)
        assert bs is None
        assert not (Path(config.out_dir) / "protrider_summary_bs.csv").exists()
        assert any("skipped" in r.message.lower() for r in caplog.records)


class TestSubsamplePlanReproducibility:
    def test_same_seed_same_plan(self):
        ids = [f"s{i}" for i in range(40)]
        config = ProtriderConfig(
            out_dir="o",
            input_intensities="d.csv",
            cohort_stability_n_runs=5,
            cohort_stability_min_runs=1,
            cohort_stability_drop_fraction=0.1,
            cohort_stability_min_samples=30,
        )
        plan_a = _generate_subsample_plan(ids, config, base_seed=42, n_drop=4)
        plan_b = _generate_subsample_plan(ids, config, base_seed=42, n_drop=4)
        assert plan_a == plan_b

    def test_different_seed_different_plan(self):
        ids = [f"s{i}" for i in range(40)]
        config = ProtriderConfig(
            out_dir="o",
            input_intensities="d.csv",
            cohort_stability_n_runs=5,
            cohort_stability_min_runs=1,
        )
        n_drop = 4
        plan_a = _generate_subsample_plan(ids, config, base_seed=1, n_drop=n_drop)
        plan_b = _generate_subsample_plan(ids, config, base_seed=2, n_drop=n_drop)
        assert plan_a != plan_b


class TestCheckpointIsolation:
    def test_iteration_uses_temp_out_dir_and_checkpoint(self, tmp_path):
        intensities_path = tmp_path / "intensities.tsv"
        samples = _make_stability_intensities(intensities_path, n_samples=35)
        config = _fast_stability_config(tmp_path, intensities_path)
        main_out = Path(config.out_dir)
        (main_out / "model.pt").write_text("fake")

        mock_result = MagicMock()
        compact = pd.DataFrame(
            [
                {
                    "sampleID": samples[0],
                    "proteinID": "PROT_000",
                    "PROTEIN_ZSCORE": 1.0,
                    "PROTEIN_PVALUE": 0.1,
                    "PROTEIN_PADJ": 0.1,
                    "PROTEIN_LOG2FC": 0.0,
                    "PROTEIN_FC": 1.0,
                    "PROTEIN_outlier": False,
                }
            ]
        )
        mock_result.to_long_df.return_value = compact

        iter_dir = tmp_path / "iter_test"
        captured = {}

        def fake_run(cfg):
            captured["out_dir"] = cfg.out_dir
            captured["checkpoint_path"] = cfg.checkpoint_path
            captured["input_intensities"] = cfg.input_intensities
            return mock_result, None, None, None

        with patch("protrider.pipeline.run", side_effect=fake_run):
            _run_single_stability_iteration(
                config=config,
                retained_sample_ids=samples[:32],
                iteration_seed=99,
                tmp_dir=iter_dir,
            )

        assert captured["out_dir"] == str(iter_dir)
        assert captured["checkpoint_path"] == str(iter_dir / "model.pt")
        assert captured["out_dir"] != config.out_dir
        assert "intensities_subset" in captured["input_intensities"]


class TestStabilitySmokeE2E:
    def test_stability_produces_bs_summary(self, tmp_path):
        intensities_path = tmp_path / "intensities.tsv"
        _make_stability_intensities(intensities_path, n_samples=35, n_proteins=20)
        config = _fast_stability_config(
            tmp_path,
            intensities_path,
            cohort_stability=True,
            cohort_stability_n_runs=2,
            cohort_stability_min_runs=2,
            cohort_stability_drop_fraction=0.1,
            cohort_stability_min_samples=30,
            cohort_stability_seed=7,
        )
        result, _, _, _ = run(config)
        result.save(config.out_dir, format="long", include_all=True)
        baseline = result.to_long_df(include_all=True)
        bs = run_cohort_stability(config, baseline)
        assert bs is not None
        out_bs = Path(config.out_dir) / "protrider_summary_bs.csv"
        bs.to_csv(out_bs, index=False)
        assert (Path(config.out_dir) / "protrider_summary.csv").exists()
        assert out_bs.exists()
        loaded = pd.read_csv(out_bs)
        required = {
            "sampleID",
            "proteinID",
            "BS_N_RUNS_REQUESTED",
            "BS_N_OBSERVED",
            "PROTEIN_outlier_call_rate",
            "PROTEIN_FC_full",
            "PROTEIN_FC_median",
        }
        assert required.issubset(loaded.columns)
        assert len(loaded) >= 1
        assert not list(Path(config.out_dir).glob("_cohort_stability_tmp/**/additional_info.csv"))
