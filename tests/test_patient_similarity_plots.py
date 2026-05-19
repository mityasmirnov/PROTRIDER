"""
Tests for patient similarity plotting helpers.
"""

import numpy as np
import pandas as pd

from protrider import plots


def _write_patient_csvs(out_dir):
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
        out_dir / "patient_similarity.csv"
    )
    pd.DataFrame(
        {
            "sampleID": samples,
            "subpopulation": ["subpopulation_1"] * 3 + ["subpopulation_2"] * 3,
            "silhouette_score_for_selected_k": [0.5] * 6,
            "selected_k": [2] * 6,
            "clustering_method": ["ward_euclidean"] * 6,
            "similarity_method": ["rbf_median_sigma"] * 6,
        }
    ).to_csv(out_dir / "patient_subpopulations.csv", index=False)
    pd.DataFrame(
        {
            "sampleID": samples,
            "PC1": np.linspace(-1, 1, 6),
            "PC2": np.linspace(1, -1, 6),
            "subpopulation": ["subpopulation_1"] * 3 + ["subpopulation_2"] * 3,
        }
    ).to_csv(out_dir / "patient_latent_pca.csv", index=False)
    subpop = ["subpopulation_1"] * 3 + ["subpopulation_2"] * 3
    pd.DataFrame(
        {
            "sampleID": samples,
            "UMAP1": np.linspace(-2, 2, 6),
            "UMAP2": np.linspace(2, -2, 6),
            "subpopulation": subpop,
        }
    ).to_csv(out_dir / "patient_latent_umap.csv", index=False)
    pd.DataFrame(
        {
            "sampleID": samples,
            "TSNE1": np.linspace(-1.5, 1.5, 6),
            "TSNE2": np.linspace(1.5, -1.5, 6),
            "subpopulation": subpop,
        }
    ).to_csv(out_dir / "patient_latent_tsne.csv", index=False)


def test_plot_patient_similarity_creates_png(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_patient_csvs(out_dir)

    result = plots.plot_patient_similarity(str(out_dir))
    png = out_dir / "plots" / "patient_similarity_heatmap.png"
    assert png.exists()
    assert result is not None


def test_plot_patient_latent_pca_creates_png(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_patient_csvs(out_dir)

    result = plots.plot_patient_latent_pca(str(out_dir))
    png = out_dir / "plots" / "patient_latent_pca.png"
    assert png.exists()
    assert result is not None


def test_plot_patient_latent_umap_creates_png(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_patient_csvs(out_dir)

    result = plots.plot_patient_latent_umap(str(out_dir))
    png = out_dir / "plots" / "patient_latent_umap.png"
    assert png.exists()
    assert result is not None


def test_plot_patient_latent_tsne_creates_png(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_patient_csvs(out_dir)

    result = plots.plot_patient_latent_tsne(str(out_dir))
    png = out_dir / "plots" / "patient_latent_tsne.png"
    assert png.exists()
    assert result is not None


def test_plots_return_none_when_files_missing(tmp_path):
    out_dir = tmp_path / "empty"
    out_dir.mkdir()

    assert plots.plot_patient_similarity(str(out_dir)) is None
    assert plots.plot_patient_latent_pca(str(out_dir)) is None
    assert plots.plot_patient_latent_umap(str(out_dir)) is None
    assert plots.plot_patient_latent_tsne(str(out_dir)) is None
