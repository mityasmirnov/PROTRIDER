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
    _subset_input_files,
    _summarize_stability,
    _write_subset_annotation,
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
    def test_single_iteration_quantiles_equal_observed_values(self):
        baseline = pd.DataFrame(
            {
                "sampleID": ["s1"],
                "proteinID": ["p1"],
                "PROTEIN_outlier": [False],
                "PROTEIN_FC": [1.0],
                "PROTEIN_LOG2FC": [0.0],
                "PROTEIN_ZSCORE": [0.0],
                "PROTEIN_PVALUE": [0.5],
                "PROTEIN_PADJ": [0.5],
            }
        )
        iter_df = pd.DataFrame(
            {
                "sampleID": ["s1"],
                "proteinID": ["p1"],
                "PROTEIN_outlier": [True],
                "PROTEIN_FC": [1.25],
                "PROTEIN_LOG2FC": [0.25],
                "PROTEIN_ZSCORE": [2.0],
                "PROTEIN_PVALUE": [0.01],
                "PROTEIN_PADJ": [0.02],
            }
        )

        out = _summarize_stability(
            [iter_df], baseline, n_runs_requested=1, n_runs_completed=1
        )

        row = out.iloc[0]
        assert row["PROTEIN_FC_q025"] == pytest.approx(1.25)
        assert row["PROTEIN_FC_q975"] == pytest.approx(1.25)
        assert row["PROTEIN_LOG2FC_q025"] == pytest.approx(0.25)
        assert row["PROTEIN_LOG2FC_q975"] == pytest.approx(0.25)

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
        out = _summarize_stability(
            [iter0, iter1], baseline, n_runs_requested=2, n_runs_completed=2
        )
        keys = set(zip(out["sampleID"], out["proteinID"]))
        assert keys == {("s1", "p1"), ("s1", "p2"), ("s1", "p3")}
        row_p1 = out[(out["sampleID"] == "s1") & (out["proteinID"] == "p1")].iloc[0]
        assert row_p1["BS_N_RUNS_REQUESTED"] == 2
        assert row_p1["BS_N_RUNS_COMPLETED"] == 2
        assert row_p1["BS_N_OBSERVED"] == 1
        assert row_p1["BS_N_MISSING"] == 1
        assert row_p1["BS_OBSERVED_FRACTION"] == 0.5
        assert row_p1["PROTEIN_outlier_call_count"] == 1
        assert row_p1["PROTEIN_outlier_call_rate"] == 1.0
        assert row_p1["PROTEIN_outlier_call_rate_all_runs"] == 0.5
        assert row_p1["PROTEIN_FC_median"] == pytest.approx(0.4)
        row_p2 = out[(out["sampleID"] == "s1") & (out["proteinID"] == "p2")].iloc[0]
        assert row_p2["BS_N_OBSERVED"] == 0
        assert row_p2["BS_N_MISSING"] == 2
        assert row_p2["BS_OBSERVED_FRACTION"] == 0.0
        assert row_p2["PROTEIN_outlier_call_count"] == 0
        assert np.isnan(row_p2["PROTEIN_outlier_call_rate"])
        assert row_p2["PROTEIN_outlier_call_rate_all_runs"] == 0.0
        assert row_p2["in_full_run"]
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

    def test_skips_when_drop_fraction_cannot_remove_samples(self, tmp_path, caplog):
        intensities_path = tmp_path / "intensities.tsv"
        _make_stability_intensities(intensities_path, n_samples=31)
        config = _fast_stability_config(
            tmp_path,
            intensities_path,
            cohort_stability=True,
            cohort_stability_n_runs=2,
            cohort_stability_min_runs=1,
            cohort_stability_drop_fraction=0.01,
            cohort_stability_min_samples=31,
        )
        baseline = pd.DataFrame(columns=["sampleID", "proteinID"])

        with caplog.at_level("WARNING"):
            bs = run_cohort_stability(config, baseline)

        assert bs is None
        assert any(
            "do not allow removing any samples" in r.message
            for r in caplog.records
        )


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
        assert captured["input_intensities"].endswith("intensities_subset.tsv")


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
            "BS_N_RUNS_COMPLETED",
            "BS_N_OBSERVED",
            "PROTEIN_outlier_call_rate",
            "PROTEIN_FC_full",
            "PROTEIN_FC_median",
        }
        assert required.issubset(loaded.columns)
        assert len(loaded) >= 1
        assert not list(Path(config.out_dir).glob("_cohort_stability_tmp/**/additional_info.csv"))


class TestStabilityBaselineIncludeAll:
    def test_stability_baseline_includes_non_outliers(self, tmp_path):
        intensities_path = tmp_path / "intensities.tsv"
        _make_stability_intensities(intensities_path, n_samples=35, n_proteins=15)
        config = _fast_stability_config(
            tmp_path,
            intensities_path,
            report_all=False,
            cohort_stability=True,
            cohort_stability_n_runs=2,
            cohort_stability_min_runs=2,
            cohort_stability_drop_fraction=0.1,
            cohort_stability_seed=3,
        )
        result, _, _, _ = run(config)
        narrow = result.to_long_df(include_all=False)
        full_baseline = result.to_long_df(include_all=True)
        assert len(full_baseline) > len(narrow)
        bs = run_cohort_stability(config, full_baseline)
        assert bs is not None
        in_full = bs[bs["in_full_run"]]
        assert in_full["PROTEIN_FC_full"].notna().any()


class TestSubsetInputs:
    def test_missing_retained_sample_id_raises(self, tmp_path):
        intensities_path = tmp_path / "intensities.tsv"
        samples = _make_stability_intensities(intensities_path, n_samples=35)
        config = _fast_stability_config(tmp_path, intensities_path)
        retained = samples[:31] + ["missing_sample"]

        with pytest.raises(ValueError, match="Retained sample IDs not found"):
            _subset_input_files(config, retained, tmp_path / "missing_subset")

    def test_parquet_input_writes_tsv_subset(self, tmp_path):
        intensities_path = tmp_path / "intensities.parquet"
        samples = _make_stability_intensities(
            tmp_path / "intensities_src.tsv", n_samples=35, n_proteins=10
        )
        pd.read_csv(tmp_path / "intensities_src.tsv", sep="\t").to_parquet(
            intensities_path, index=False
        )
        config = _fast_stability_config(tmp_path, intensities_path)
        subset_dir = tmp_path / "subset"
        out_int, _ = _subset_input_files(config, samples[:32], subset_dir)
        assert out_int.endswith("intensities_subset.tsv")
        assert Path(out_int).exists()
        head = pd.read_csv(out_int, sep="\t", nrows=1)
        assert "protein_ID" in head.columns

    def test_mixed_delimiter_annotation(self, tmp_path):
        samples = [f"sample_{i}" for i in range(35)]
        int_tsv = tmp_path / "intensities.tsv"
        _make_stability_intensities(int_tsv, n_samples=35, n_proteins=8)
        anno_csv = tmp_path / "annotation.csv"
        pd.DataFrame({"sampleID": samples, "batch": ["A"] * len(samples)}).to_csv(
            anno_csv, index=False
        )
        config = _fast_stability_config(
            tmp_path, int_tsv, sample_annotation=str(anno_csv)
        )
        out_path = _write_subset_annotation(config, samples[:32], tmp_path / "sub_a")
        written = pd.read_csv(out_path)
        assert len(written) == 32
        assert list(written.columns) == ["sampleID", "batch"]

        int_csv = tmp_path / "intensities.csv"
        df = pd.read_csv(int_tsv, sep="\t")
        df.to_csv(int_csv, index=False)
        anno_tsv = tmp_path / "annotation.tsv"
        pd.DataFrame({"sampleID": samples, "batch": ["B"] * len(samples)}).to_csv(
            anno_tsv, sep="\t", index=False
        )
        config2 = _fast_stability_config(
            tmp_path, int_csv, sample_annotation=str(anno_tsv)
        )
        out_path2 = _write_subset_annotation(config2, samples[:30], tmp_path / "sub_b")
        written2 = pd.read_csv(out_path2, sep="\t")
        assert len(written2) == 30

    def test_numeric_sample_ids_proteins_as_columns(self, tmp_path):
        rng = np.random.default_rng(0)
        n_samples, n_proteins = 35, 8
        sample_ids = list(range(100, 100 + n_samples))
        proteins = [f"PROT_{i:03d}" for i in range(n_proteins)]
        values = rng.uniform(50, 200, size=(n_samples, n_proteins)) * 1000.0
        int_path = tmp_path / "intensities_numeric_samples.tsv"
        df = pd.DataFrame(values, index=sample_ids, columns=proteins)
        df.index.name = "sampleID"
        df = df.reset_index()
        df.to_csv(int_path, sep="\t", index=False)

        config = _fast_stability_config(
            tmp_path,
            int_path,
            input_format="proteins_as_columns",
            index_col="sampleID",
        )
        retained = [str(sid) for sid in sample_ids[:32]]
        out_int, _ = _subset_input_files(config, retained, tmp_path / "subset_num")
        written = pd.read_csv(out_int, sep="\t")
        assert len(written) == 32
        assert set(written["sampleID"].astype(str)) == set(retained)

        anno_path = tmp_path / "annotation_numeric.csv"
        pd.DataFrame(
            {"sampleID": sample_ids, "batch": ["A"] * n_samples}
        ).to_csv(anno_path, index=False)
        config_anno = _fast_stability_config(
            tmp_path,
            int_path,
            input_format="proteins_as_columns",
            index_col="sampleID",
            sample_annotation=str(anno_path),
        )
        out_anno = _write_subset_annotation(
            config_anno, retained, tmp_path / "anno_num"
        )
        anno_written = pd.read_csv(out_anno)
        assert len(anno_written) == 32
        assert set(anno_written["sampleID"].astype(str)) == set(retained)


class TestRuntimeBudgetDenominator:
    def test_completed_runs_used_when_budget_stops_early(self):
        baseline = pd.DataFrame(
            {
                "sampleID": ["s1"],
                "proteinID": ["p1"],
                "PROTEIN_outlier": [False],
                "PROTEIN_FC": [1.0],
                "PROTEIN_LOG2FC": [0.0],
                "PROTEIN_ZSCORE": [0.0],
                "PROTEIN_PVALUE": [0.5],
                "PROTEIN_PADJ": [0.5],
            }
        )
        iter_df = pd.DataFrame(
            {
                "sampleID": ["s1"],
                "proteinID": ["p1"],
                "PROTEIN_outlier": [True],
                "PROTEIN_FC": [1.1],
                "PROTEIN_LOG2FC": [0.1],
                "PROTEIN_ZSCORE": [1.0],
                "PROTEIN_PVALUE": [0.01],
                "PROTEIN_PADJ": [0.05],
            }
        )
        out = _summarize_stability(
            [iter_df],
            baseline,
            n_runs_requested=10,
            n_runs_completed=1,
        )
        row = out.iloc[0]
        assert row["BS_N_RUNS_REQUESTED"] == 10
        assert row["BS_N_RUNS_COMPLETED"] == 1
        assert row["BS_N_OBSERVED"] == 1
        assert row["BS_N_MISSING"] == 0
        assert row["PROTEIN_outlier_call_rate_all_runs"] == 1.0
