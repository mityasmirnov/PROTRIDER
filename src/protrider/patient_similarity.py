from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

__all__ = ["PatientSimilarity", "compute_patient_similarity"]

SIMILARITY_METHOD = "rbf_median_sigma"
DISTANCE_METRIC = "euclidean"
LATENT_SCALING = "standard_scaler"
CLUSTERING_METHOD = "ward_euclidean"
UMAP_METHOD = "umap_euclidean"
TSNE_METHOD = "sklearn_tsne_euclidean"


def _subpopulation_labels(
    sample_index: pd.Index,
    subpopulations: pd.DataFrame,
) -> list[str]:
    subpopulation_map = subpopulations.set_index("sampleID")["subpopulation"]
    return [subpopulation_map.get(sid, "subpopulation_1") for sid in sample_index]


def _agglomerative_cluster(n_clusters: int) -> AgglomerativeClustering:
    """Construct AgglomerativeClustering with sklearn version compatibility."""
    try:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
            metric="euclidean",
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
            affinity="euclidean",
        )


def _labels_to_subpopulation_names(labels: np.ndarray) -> list[str]:
    """Map cluster ids to stable subpopulation_1, subpopulation_2, ... by cluster size."""
    unique_labels = np.unique(labels)
    sizes = [(label, int(np.sum(labels == label))) for label in unique_labels]
    sizes.sort(key=lambda item: (-item[1], item[0]))
    label_to_name = {
        label: f"subpopulation_{rank + 1}" for rank, (label, _) in enumerate(sizes)
    }
    return [label_to_name[label] for label in labels]


@dataclass
class PatientSimilarity:
    """Patient/sample similarity and subpopulation summaries from latent embeddings."""

    similarity: pd.DataFrame
    subpopulations: Optional[pd.DataFrame] = None
    pca_coordinates: Optional[pd.DataFrame] = None
    umap_coordinates: Optional[pd.DataFrame] = None
    tsne_coordinates: Optional[pd.DataFrame] = None
    info: Optional[pd.DataFrame] = None

    def save(self, out_dir: str) -> dict[str, Path]:
        """Write patient similarity outputs to out_dir and return paths written."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        similarity_path = out_path / "patient_similarity.csv"
        self.similarity.to_csv(similarity_path, header=True, index=True)
        written["patient_similarity"] = similarity_path
        logger.info("Saved patient similarity matrix to %s", similarity_path)

        if self.subpopulations is not None:
            subpop_path = out_path / "patient_subpopulations.csv"
            self.subpopulations.to_csv(subpop_path, header=True, index=False)
            written["patient_subpopulations"] = subpop_path
            logger.info("Saved patient subpopulations to %s", subpop_path)

        if self.pca_coordinates is not None:
            pca_path = out_path / "patient_latent_pca.csv"
            self.pca_coordinates.to_csv(pca_path, header=True, index=False)
            written["patient_latent_pca"] = pca_path
            logger.info("Saved patient latent PCA coordinates to %s", pca_path)

        if self.umap_coordinates is not None:
            umap_path = out_path / "patient_latent_umap.csv"
            self.umap_coordinates.to_csv(umap_path, header=True, index=False)
            written["patient_latent_umap"] = umap_path
            logger.info("Saved patient latent UMAP coordinates to %s", umap_path)

        if self.tsne_coordinates is not None:
            tsne_path = out_path / "patient_latent_tsne.csv"
            self.tsne_coordinates.to_csv(tsne_path, header=True, index=False)
            written["patient_latent_tsne"] = tsne_path
            logger.info("Saved patient latent t-SNE coordinates to %s", tsne_path)

        if self.info is not None:
            info_path = out_path / "patient_similarity_info.csv"
            self.info.to_csv(info_path, header=True, index=False)
            written["patient_similarity_info"] = info_path
            logger.info("Saved patient similarity metadata to %s", info_path)

        return written


def _cluster_subpopulations(
    z_scaled: np.ndarray,
    sample_index: pd.Index,
    max_clusters: int,
    min_samples_for_clustering: int,
    clustering_status: str,
) -> tuple[pd.DataFrame, int, float, str]:
    """
    Cluster standardized latent coordinates and select k by silhouette score.

    For small cohorts (n < min_samples_for_clustering), all samples are assigned
    to subpopulation_1 without attempting automatic k selection.
    """
    n_samples = z_scaled.shape[0]
    base_record = {
        "sampleID": list(sample_index),
        "clustering_method": CLUSTERING_METHOD,
        "similarity_method": SIMILARITY_METHOD,
    }

    if n_samples < min_samples_for_clustering:
        return (
            pd.DataFrame(
                {
                    **base_record,
                    "subpopulation": ["subpopulation_1"] * n_samples,
                    "silhouette_score_for_selected_k": [np.nan] * n_samples,
                    "selected_k": [1] * n_samples,
                }
            ),
            1,
            np.nan,
            "not_enough_samples_for_clustering",
        )

    max_k = min(max_clusters, n_samples - 1)
    if max_k < 2:
        return (
            pd.DataFrame(
                {
                    **base_record,
                    "subpopulation": ["subpopulation_1"] * n_samples,
                    "silhouette_score_for_selected_k": [np.nan] * n_samples,
                    "selected_k": [1] * n_samples,
                }
            ),
            1,
            np.nan,
            "not_enough_samples_for_clustering",
        )

    best_k = 2
    best_score = -1.0
    best_labels = np.zeros(n_samples, dtype=int)
    clustering_failed = False

    for k in range(2, max_k + 1):
        try:
            model = _agglomerative_cluster(k)
            labels = model.fit_predict(z_scaled)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(z_scaled, labels))
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception as exc:
            logger.warning("Subpopulation clustering failed for k=%s: %s", k, exc)
            clustering_failed = True

    if len(np.unique(best_labels)) < 2:
        subpopulation_names = ["subpopulation_1"] * n_samples
        status = "clustering_degenerate_single_cluster"
        selected_k = 1
        selected_silhouette = np.nan
    else:
        subpopulation_names = _labels_to_subpopulation_names(best_labels)
        status = "clustering_failed" if clustering_failed else clustering_status
        selected_k = best_k
        selected_silhouette = best_score

    return (
        pd.DataFrame(
            {
                **base_record,
                "subpopulation": subpopulation_names,
                "silhouette_score_for_selected_k": [selected_silhouette] * n_samples,
                "selected_k": [selected_k] * n_samples,
            }
        ),
        selected_k,
        selected_silhouette,
        status,
    )


def _compute_pca_coordinates(
    z_scaled: np.ndarray,
    sample_index: pd.Index,
    subpopulations: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Project standardized latent coordinates to 2D with PCA for visualization."""
    n_samples, n_features = z_scaled.shape
    if n_samples < 2:
        return None

    if n_features == 1:
        pc1 = z_scaled[:, 0]
        pc2 = np.zeros(n_samples)
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(z_scaled)
        pc1 = coords[:, 0]
        pc2 = coords[:, 1]

    subpopulation = _subpopulation_labels(sample_index, subpopulations)

    return pd.DataFrame(
        {
            "sampleID": list(sample_index),
            "PC1": pc1,
            "PC2": pc2,
            "subpopulation": subpopulation,
        }
    )


def _compute_umap_coordinates(
    z_scaled: np.ndarray,
    sample_index: pd.Index,
    subpopulations: pd.DataFrame,
    random_state: int = 42,
) -> tuple[Optional[pd.DataFrame], str]:
    """Project standardized latents to 2D with UMAP for visualization."""
    n_samples = z_scaled.shape[0]
    if n_samples < 3:
        logger.warning(
            "Skipping UMAP projection: need at least 3 samples (got %s)", n_samples
        )
        return None, "skipped_insufficient_samples"

    try:
        from umap import UMAP
    except ImportError:
        logger.warning("Skipping UMAP projection: umap-learn is not installed")
        return None, "skipped_import"

    n_neighbors = min(15, max(2, n_samples - 1))
    try:
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=random_state,
        )
        coords = reducer.fit_transform(z_scaled)
    except Exception as exc:
        logger.warning("Skipping UMAP projection after failure: %s", exc)
        return None, "failed"

    subpopulation = _subpopulation_labels(sample_index, subpopulations)
    return (
        pd.DataFrame(
            {
                "sampleID": list(sample_index),
                "UMAP1": coords[:, 0],
                "UMAP2": coords[:, 1],
                "subpopulation": subpopulation,
            }
        ),
        "ok",
    )


def _compute_tsne_coordinates(
    z_scaled: np.ndarray,
    sample_index: pd.Index,
    subpopulations: pd.DataFrame,
    random_state: int = 42,
) -> tuple[Optional[pd.DataFrame], str]:
    """Project standardized latents to 2D with t-SNE for visualization."""
    from sklearn.manifold import TSNE

    n_samples = z_scaled.shape[0]
    if n_samples < 4:
        logger.warning(
            "Skipping t-SNE projection: need at least 4 samples (got %s)", n_samples
        )
        return None, "skipped_insufficient_samples"

    perplexity = min(30, max(2, (n_samples - 1) // 3))
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric="euclidean",
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        coords = tsne.fit_transform(z_scaled)
    except Exception as exc:
        logger.warning("Skipping t-SNE projection after failure: %s", exc)
        return None, "failed"

    subpopulation = _subpopulation_labels(sample_index, subpopulations)
    return (
        pd.DataFrame(
            {
                "sampleID": list(sample_index),
                "TSNE1": coords[:, 0],
                "TSNE2": coords[:, 1],
                "subpopulation": subpopulation,
            }
        ),
        "ok",
    )


def compute_patient_similarity(
    latent_samples: pd.DataFrame,
    max_clusters: int = 10,
    min_samples_for_clustering: int = 4,
) -> Optional[PatientSimilarity]:
    """
    Compute patient/sample similarity and subpopulations from latent embeddings.

    Latent dimensions are standardized so no single axis dominates Euclidean
    distances. Pairwise distances are converted to RBF similarities in [0, 1]
    using the median nonzero distance as the kernel width (sigma).
    """
    z = latent_samples.copy()
    z = z.apply(pd.to_numeric, errors="coerce")
    z = z.replace([np.inf, -np.inf], np.nan)

    valid = z.notna().all(axis=1)
    if not valid.all():
        dropped = z.index[~valid].tolist()
        logger.warning(
            "Dropping samples with non-finite latent values before patient similarity: %s",
            dropped,
        )
        z = z.loc[valid]

    if z.shape[0] < 2:
        logger.warning(
            "Skipping patient similarity: fewer than two valid samples (%s)",
            z.shape[0],
        )
        return None

    # Standardize latent axes before distance-based similarity.
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z.values)

    d = pairwise_distances(z_scaled, metric=DISTANCE_METRIC)
    nonzero = d[d > 0]
    if nonzero.size == 0:
        logger.warning(
            "All latent samples are identical; using identity patient similarity matrix"
        )
        similarity_array = np.eye(z.shape[0])
        status = "degenerate_all_samples_identical"
        sigma = np.nan
    else:
        sigma = float(np.median(nonzero))
        similarity_array = np.exp(-(d ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(similarity_array, 1.0)
        status = "ok"

    similarity = pd.DataFrame(
        similarity_array,
        index=z.index,
        columns=z.index,
    )

    subpopulations, selected_k, selected_silhouette, cluster_status = _cluster_subpopulations(
        z_scaled,
        z.index,
        max_clusters=max_clusters,
        min_samples_for_clustering=min_samples_for_clustering,
        clustering_status=status,
    )
    pca_coordinates = _compute_pca_coordinates(z_scaled, z.index, subpopulations)
    if nonzero.size == 0:
        umap_coordinates, umap_status = None, "skipped_degenerate_latents"
        tsne_coordinates, tsne_status = None, "skipped_degenerate_latents"
    else:
        umap_coordinates, umap_status = _compute_umap_coordinates(
            z_scaled, z.index, subpopulations
        )
        tsne_coordinates, tsne_status = _compute_tsne_coordinates(
            z_scaled, z.index, subpopulations
        )

    info = pd.DataFrame(
        [
            {
                "similarity_method": SIMILARITY_METHOD,
                "distance_metric": DISTANCE_METRIC,
                "latent_scaling": LATENT_SCALING,
                "sigma": sigma,
                "n_samples": z.shape[0],
                "q": z.shape[1],
                "clustering_method": CLUSTERING_METHOD,
                "selected_k": selected_k,
                "selected_silhouette_score": selected_silhouette,
                "status": cluster_status,
                "umap_method": UMAP_METHOD if umap_coordinates is not None else np.nan,
                "umap_status": umap_status,
                "tsne_method": TSNE_METHOD if tsne_coordinates is not None else np.nan,
                "tsne_status": tsne_status,
            }
        ]
    )

    return PatientSimilarity(
        similarity=similarity,
        subpopulations=subpopulations,
        pca_coordinates=pca_coordinates,
        umap_coordinates=umap_coordinates,
        tsne_coordinates=tsne_coordinates,
        info=info,
    )
