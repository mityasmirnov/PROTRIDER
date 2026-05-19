"""
Tests for co-outlier patient stratification plotting helpers.
"""

import numpy as np
import pandas as pd

from protrider import plots


def _write_cooutlier_csvs(out_dir):
    samples = [f"s{i}" for i in range(6)]
    sim = np.eye(6)
    for i in range(3):
        for j in range(i + 1, 3):
            sim[i, j] = sim[j, i] = 0.9
    for i in range(3, 6):
        for j in range(i + 1, 6):
            sim[i, j] = sim[j, i] = 0.85
    for i in range(3):
        for j in range(3, 6):
            sim[i, j] = sim[j, i] = 0.2

    pd.DataFrame(sim, index=samples, columns=samples).to_csv(
        out_dir / "cooutlier_patient_similarity.csv"
    )
    subpop = ["subpopulation_1"] * 3 + ["subpopulation_2"] * 3
    pd.DataFrame(
        {
            "sampleID": samples,
            "subpopulation": subpop,
            "n_up": [2, 2, 2, 0, 0, 0],
            "n_down": [0, 0, 0, 2, 2, 2],
            "n_total": [2, 2, 2, 2, 2, 2],
            "eligible_for_clustering": [True] * 6,
            "selected_k": [2] * 6,
            "silhouette_score_for_selected_k": [0.4] * 6,
            "clustering_method": ["average_agglomerative_precomputed_jaccard_distance"] * 6,
            "similarity_method": ["directional_jaccard_same_direction"] * 6,
            "z_threshold": [3.0] * 6,
            "status": ["ok"] * 6,
        }
    ).to_csv(out_dir / "cooutlier_patient_subpopulations.csv", index=False)

    for name, x_col, y_col in [
        ("cooutlier_patient_pca.csv", "PC1", "PC2"),
        ("cooutlier_patient_umap.csv", "UMAP1", "UMAP2"),
        ("cooutlier_patient_tsne.csv", "TSNE1", "TSNE2"),
    ]:
        pd.DataFrame(
            {
                "sampleID": samples,
                x_col: np.linspace(-1, 1, 6),
                y_col: np.linspace(1, -1, 6),
                "subpopulation": subpop,
            }
        ).to_csv(out_dir / name, index=False)


def test_plot_cooutlier_patient_similarity_creates_png(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_cooutlier_csvs(out_dir)

    result = plots.plot_cooutlier_patient_similarity(str(out_dir))
    png = out_dir / "plots" / "cooutlier_patient_similarity_heatmap.png"
    assert png.exists()
    assert result is not None


def test_plot_cooutlier_embeddings_create_png(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_cooutlier_csvs(out_dir)

    assert plots.plot_cooutlier_patient_pca(str(out_dir)) is not None
    assert (out_dir / "plots" / "cooutlier_patient_pca.png").exists()

    assert plots.plot_cooutlier_patient_umap(str(out_dir)) is not None
    assert (out_dir / "plots" / "cooutlier_patient_umap.png").exists()

    assert plots.plot_cooutlier_patient_tsne(str(out_dir)) is not None
    assert (out_dir / "plots" / "cooutlier_patient_tsne.png").exists()


def test_cooutlier_plots_missing_files_return_none(tmp_path):
    out_dir = tmp_path / "empty"
    out_dir.mkdir()

    assert plots.plot_cooutlier_patient_similarity(str(out_dir)) is None
    assert plots.plot_cooutlier_patient_pca(str(out_dir)) is None
    assert plots.plot_cooutlier_patient_umap(str(out_dir)) is None
    assert plots.plot_cooutlier_patient_tsne(str(out_dir)) is None
