#!/usr/bin/env python3
"""
End-to-end PROTRIDER run on omicsDiagnostics real data.

Exercises cohort stability, patient embeddings, co-outlier analysis, and all
plot CLI commands. Intended for M1 Mac (device=cpu).

Usage:
  conda run -n omicsDiagnosticsDev python tests/run_oht_n_pro_real.py
  conda run -n omicsDiagnosticsDev python tests/run_oht_n_pro_real.py --smoke
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "rwd" / "config_protrider_dev.yml"
DEFAULT_INTENSITIES = Path(
    "/Users/Mitya/Desktop/working/omicsDagnostics_data/processed_data/protrider/protrider_data.tsv"
)
DEFAULT_ANNOTATION = Path(
    "/Users/Mitya/Desktop/working/omicsDagnostics_data/processed_data/protrider/protrider_annotation.tsv"
)
DEFAULT_OUT_DIR = Path("/Volumes/Transcend/prot/protrider_stability_dev")


def build_run_config(
    base_config_path: Path,
    out_dir: Path,
    *,
    smoke: bool,
) -> Path:
    config = yaml.safe_load(base_config_path.read_text())

    config["out_dir"] = str(out_dir)
    config["input_intensities"] = str(DEFAULT_INTENSITIES)
    config["sample_annotation"] = str(DEFAULT_ANNOTATION)
    config["index_col"] = "geneID"
    config["input_format"] = "proteins_as_rows"
    config["find_q_method"] = "OHT"
    config["report_all"] = True
    config["device"] = "cpu"
    config["n_jobs"] = -1
    config["checkpoint_path"] = None

    config["cohort_stability"] = True
    config["cohort_stability_require_oht"] = True
    config["cohort_stability_save_iteration_files"] = False
    config["cohort_stability_drop_fraction"] = 0.1
    config["cohort_stability_min_samples"] = 30
    config["cohort_stability_seed"] = 42

    config["export_latent_space"] = True
    config["export_patient_similarity"] = True
    config["export_cooutlier_patient_similarity"] = True

    if smoke:
        config["n_epochs"] = 5
        config["patience"] = 3
        config["cohort_stability_n_runs"] = 2
        config["cohort_stability_min_runs"] = 2
    else:
        config["cohort_stability_n_runs"] = 10
        config["cohort_stability_min_runs"] = 10

    out_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = out_dir / "config_protrider_stability_run.yaml"
    run_config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    return run_config_path


def protrider_bin() -> str:
    import shutil

    for candidate in (
        shutil.which("protrider"),
        "/opt/miniconda3/envs/omicsDiagnosticsDev/bin/protrider",
    ):
        if candidate:
            return candidate
    raise FileNotFoundError("protrider CLI not found; activate omicsDiagnosticsDev")


def run_cmd(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def assert_outputs(out_dir: Path) -> None:
    required = [
        "protrider_summary.csv",
        "protrider_summary_bs.csv",
        "latent_samples.csv",
        "patient_subpopulations.csv",
        "cooutlier_patient_subpopulations.csv",
    ]
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected outputs in {out_dir}: {missing}")

    plot_dir = out_dir / "plots"
    expected_plots = [
        "patient_similarity_heatmap.png",
        "patient_latent_pca.png",
        "patient_latent_umap.png",
        "patient_latent_tsne.png",
        "cooutlier_patient_similarity_heatmap.png",
        "cooutlier_patient_pca.png",
        "cooutlier_patient_umap.png",
        "cooutlier_patient_tsne.png",
        "pvalues_dist.png",
        "qqplots.png",
        "aberrant_per_sample.png",
        "training_loss.png",
    ]
    missing_plots = [name for name in expected_plots if not (plot_dir / name).exists()]
    if missing_plots:
        raise FileNotFoundError(f"Missing plots in {plot_dir}: {missing_plots}")

    print(f"\nAll outputs verified under {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast validation: 5 epochs, 2 stability runs",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before run",
    )
    args = parser.parse_args()

    for path, label in [
        (args.base_config, "base config"),
        (DEFAULT_INTENSITIES, "intensities"),
        (DEFAULT_ANNOTATION, "annotation"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir, ignore_errors=True)

    run_config = build_run_config(args.base_config, args.out_dir, smoke=args.smoke)

    cli = protrider_bin()
    run_cmd([cli, "run", "--config", str(run_config)])
    run_cmd([cli, "plot", "--config", str(run_config), "all"])

    assert_outputs(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
