from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

__all__ = ["CoOutlierSimilarity", "compute_cooutlier_similarity"]

SIMILARITY_METHOD = "directional_jaccard_same_direction"
CLUSTERING_METHOD = "average_agglomerative_precomputed_jaccard_distance"
PCA_METHOD = "truncated_svd_sparse_cooutlier_profile"
UMAP_METHOD = "umap_precomputed_jaccard_distance"
TSNE_METHOD = "sklearn_tsne_precomputed_jaccard_distance"
LOW_BURDEN_LABEL = "unclassified_low_burden"


def _subpopulation_labels(
    sample_index: pd.Index,
    subpopulations: pd.DataFrame,
) -> list[str]:
    subpopulation_map = subpopulations.set_index("sampleID")["subpopulation"]
    return [subpopulation_map.get(sid, LOW_BURDEN_LABEL) for sid in sample_index]


def _agglomerative_precomputed(n_clusters: int) -> AgglomerativeClustering:
    """Average-linkage clustering on a precomputed distance matrix (not Ward/Euclidean)."""
    try:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="average",
            metric="precomputed",
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="average",
            affinity="precomputed",
        )


def _labels_to_subpopulation_names(labels: np.ndarray) -> list[str]:
    """Map cluster ids to stable subpopulation_1, subpopulation_2, ... by cluster size."""
    unique_labels = np.unique(labels)
    sizes = [(label, int(np.sum(labels == label))) for label in unique_labels]
    sizes.sort(key=lambda item: (-item[1], item[0]))
    label_to_name = {
        label: f"subpopulation_{rank + 1}"
        for rank, (label, _) in enumerate(sizes)
    }
    return [label_to_name[label] for label in labels]


def _compute_directional_jaccard(
    zscores: pd.DataFrame,
    z_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Directional Jaccard similarity from sparse up/down outlier indicators.

    Up and down are separated so opposite-direction aberrations on the same protein
    never count as a match. Sparse matrix multiplication counts same-direction
    intersections efficiently.
    """
    z = zscores.copy()
    z = z.apply(pd.to_numeric, errors="coerce")
    z = z.replace([np.inf, -np.inf], np.nan)

    sample_names = z.index
    z_arr = z.to_numpy(dtype=float, copy=True)

    # NaN comparisons are False; non-aberrant proteins do not contribute.
    is_up = z_arr >= z_threshold
    is_down = z_arr <= -z_threshold

    b_up = sparse.csr_matrix(is_up.astype(np.int8))
    b_down = sparse.csr_matrix(is_down.astype(np.int8))

    c_up = b_up @ b_up.T
    c_down = b_down @ b_down.T
    intersection = (c_up + c_down).toarray().astype(float)

    burden_up = np.asarray(b_up.sum(axis=1)).ravel().astype(int)
    burden_down = np.asarray(b_down.sum(axis=1)).ravel().astype(int)
    burden_total = burden_up + burden_down

    denom = burden_total[:, None] + burden_total[None, :] - intersection
    similarity = np.zeros_like(intersection, dtype=float)
    valid = denom > 0
    similarity[valid] = intersection[valid] / denom[valid]

    # Self-similarity is 1 only when the sample has at least one aberrant protein.
    np.fill_diagonal(similarity, np.where(burden_total > 0, 1.0, 0.0))

    sim_df = pd.DataFrame(
        similarity,
        index=sample_names,
        columns=sample_names,
    )
    burden_df = pd.DataFrame(
        {
            "sampleID": sample_names,
            "n_up": burden_up,
            "n_down": burden_down,
            "n_total": burden_total,
        }
    )
    return sim_df, burden_df, b_up, b_down


def _cluster_cooutlier_subpopulations(
    similarity: pd.DataFrame,
    burden: pd.DataFrame,
    z_threshold: float,
    min_anomalies: int,
    max_clusters: int,
    min_samples_for_clustering: int,
) -> tuple[pd.DataFrame, int, float, str]:
    """Cluster on Jaccard distance (1 - similarity); low-burden samples stay unclassified."""
    sample_ids = list(similarity.index)
    burden_map = burden.set_index("sampleID")
    eligible_mask = burden_map.loc[sample_ids, "n_total"].values >= min_anomalies
    eligible_ids = [sid for sid, ok in zip(sample_ids, eligible_mask) if ok]

    base_cols = {
        "sampleID": sample_ids,
        "n_up": [int(burden_map.loc[sid, "n_up"]) for sid in sample_ids],
        "n_down": [int(burden_map.loc[sid, "n_down"]) for sid in sample_ids],
        "n_total": [int(burden_map.loc[sid, "n_total"]) for sid in sample_ids],
        "eligible_for_clustering": eligible_mask.tolist(),
        "clustering_method": [CLUSTERING_METHOD] * len(sample_ids),
        "similarity_method": [SIMILARITY_METHOD] * len(sample_ids),
        "z_threshold": [z_threshold] * len(sample_ids),
    }

    n_eligible = len(eligible_ids)
    if n_eligible == 0:
        return (
            pd.DataFrame(
                {
                    **base_cols,
                    "subpopulation": [LOW_BURDEN_LABEL] * len(sample_ids),
                    "selected_k": [np.nan] * len(sample_ids),
                    "silhouette_score_for_selected_k": [np.nan] * len(sample_ids),
                    "status": ["no_eligible_samples"] * len(sample_ids),
                }
            ),
            0,
            np.nan,
            "no_eligible_samples",
        )

    subpopulation = [LOW_BURDEN_LABEL if not ok else "" for ok in eligible_mask]
    selected_k = np.nan
    selected_silhouette = np.nan
    cluster_status = "ok"

    if n_eligible < min_samples_for_clustering:
        for i, sid in enumerate(sample_ids):
            if sid in eligible_ids:
                subpopulation[i] = "subpopulation_1"
        cluster_status = "not_enough_samples_for_clustering"
        status_col = [
            cluster_status if eligible_mask[i] else "unclassified_low_burden"
            for i in range(len(sample_ids))
        ]
        return (
            pd.DataFrame(
                {
                    **base_cols,
                    "subpopulation": subpopulation,
                    "selected_k": [1 if sid in eligible_ids else np.nan for sid in sample_ids],
                    "silhouette_score_for_selected_k": [selected_silhouette] * len(sample_ids),
                    "status": status_col,
                }
            ),
            1,
            selected_silhouette,
            cluster_status,
        )

    sim_eligible = similarity.loc[eligible_ids, eligible_ids].values.astype(float)
    distance = 1.0 - sim_eligible
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)

    max_k = min(max_clusters, n_eligible - 1)
    if max_k < 2:
        for i, sid in enumerate(sample_ids):
            if sid in eligible_ids:
                subpopulation[i] = "subpopulation_1"
        cluster_status = "not_enough_samples_for_clustering"
        status_col = [
            cluster_status if eligible_mask[i] else "unclassified_low_burden"
            for i in range(len(sample_ids))
        ]
        return (
            pd.DataFrame(
                {
                    **base_cols,
                    "subpopulation": subpopulation,
                    "selected_k": [1 if sid in eligible_ids else np.nan for sid in sample_ids],
                    "silhouette_score_for_selected_k": [np.nan] * len(sample_ids),
                    "status": status_col,
                }
            ),
            1,
            np.nan,
            cluster_status,
        )

    best_k = 2
    best_score = -1.0
    best_labels = np.zeros(n_eligible, dtype=int)
    clustering_failed = False

    for k in range(2, max_k + 1):
        try:
            model = _agglomerative_precomputed(k)
            labels = model.fit_predict(distance)
            if len(np.unique(labels)) < 2:
                continue
            score = float(
                silhouette_score(distance, labels, metric="precomputed")
            )
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception as exc:
            logger.warning("Co-outlier clustering failed for k=%s: %s", k, exc)
            clustering_failed = True

    if len(np.unique(best_labels)) < 2:
        eligible_names = ["subpopulation_1"] * n_eligible
        cluster_status = "clustering_degenerate_single_cluster"
        selected_k = 1
        selected_silhouette = np.nan
    else:
        eligible_names = _labels_to_subpopulation_names(best_labels)
        cluster_status = "clustering_failed" if clustering_failed else "ok"
        selected_k = best_k
        selected_silhouette = best_score

    eligible_lookup = dict(zip(eligible_ids, eligible_names))
    for i, sid in enumerate(sample_ids):
        if sid in eligible_lookup:
            subpopulation[i] = eligible_lookup[sid]

    status_col = [
        "unclassified_low_burden" if not eligible_mask[i] else cluster_status
        for i in range(len(sample_ids))
    ]

    return (
        pd.DataFrame(
            {
                **base_cols,
                "subpopulation": subpopulation,
                "selected_k": [
                    selected_k if eligible_mask[i] else np.nan
                    for i in range(len(sample_ids))
                ],
                "silhouette_score_for_selected_k": [
                    selected_silhouette if eligible_mask[i] else np.nan
                    for i in range(len(sample_ids))
                ],
                "status": status_col,
            }
        ),
        int(selected_k) if not np.isnan(selected_k) else 0,
        selected_silhouette,
        cluster_status,
    )


def _sparse_cooutlier_features(
    b_up: sparse.csr_matrix,
    b_down: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Concatenate up/down binary indicators (samples × 2·proteins) for projection."""
    return sparse.hstack([b_up, b_down], format="csr")


def _projection_frame(
    sample_ids: list,
    subpopulations: pd.DataFrame,
    x_name: str,
    y_name: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> pd.DataFrame:
    subpopulation = _subpopulation_labels(pd.Index(sample_ids), subpopulations)
    return pd.DataFrame(
        {
            "sampleID": sample_ids,
            x_name: x_vals,
            y_name: y_vals,
            "subpopulation": subpopulation,
        }
    )


def _nan_projection_frame(
    sample_ids: list,
    subpopulations: pd.DataFrame,
    x_name: str,
    y_name: str,
) -> pd.DataFrame:
    return _projection_frame(
        sample_ids,
        subpopulations,
        x_name,
        y_name,
        np.full(len(sample_ids), np.nan),
        np.full(len(sample_ids), np.nan),
    )


def _compute_pca_coordinates(
    features: sparse.csr_matrix,
    sample_ids: list,
    eligible_mask: np.ndarray,
    subpopulations: pd.DataFrame,
    random_state: int,
) -> tuple[Optional[pd.DataFrame], str]:
    """TruncatedSVD (PCA-like) on sparse co-outlier profile; ineligible samples get NaN coords."""
    n_samples = features.shape[0]
    if n_samples < 2:
        return None, "skipped_insufficient_samples"

    n_eligible = int(np.sum(eligible_mask))
    if n_eligible < 2:
        return (
            _nan_projection_frame(sample_ids, subpopulations, "PC1", "PC2"),
            "skipped_insufficient_eligible_samples",
        )

    try:
        svd = TruncatedSVD(n_components=2, random_state=random_state)
        svd.fit(features[eligible_mask])
        coords = np.full((n_samples, 2), np.nan, dtype=float)
        coords[eligible_mask] = svd.transform(features[eligible_mask])
    except Exception as exc:
        logger.warning("Skipping co-outlier PCA-like projection: %s", exc)
        return None, "failed"

    return (
        _projection_frame(
            sample_ids, subpopulations, "PC1", "PC2", coords[:, 0], coords[:, 1]
        ),
        "ok",
    )


def _compute_umap_coordinates(
    similarity: pd.DataFrame,
    sample_ids: list,
    eligible_mask: np.ndarray,
    subpopulations: pd.DataFrame,
    random_state: int,
) -> tuple[Optional[pd.DataFrame], str]:
    n_samples = len(sample_ids)
    if n_samples < 3:
        logger.warning(
            "Skipping co-outlier UMAP: need at least 3 samples (got %s)", n_samples
        )
        return None, "skipped_insufficient_samples"

    n_eligible = int(np.sum(eligible_mask))
    if n_eligible < 3:
        return (
            _nan_projection_frame(sample_ids, subpopulations, "UMAP1", "UMAP2"),
            "skipped_insufficient_eligible_samples",
        )

    try:
        from umap import UMAP
    except ImportError:
        logger.warning("Skipping co-outlier UMAP: umap-learn is not installed")
        return None, "skipped_import"

    eligible_ids = [sid for sid, ok in zip(sample_ids, eligible_mask) if ok]
    sim_eligible = similarity.loc[eligible_ids, eligible_ids].values.astype(float)
    distance = 1.0 - sim_eligible
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)

    n_neighbors = min(15, max(2, n_eligible - 1))
    try:
        reducer = UMAP(
            n_components=2,
            metric="precomputed",
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=random_state,
        )
        coords_eligible = reducer.fit_transform(distance)
        coords = np.full((n_samples, 2), np.nan, dtype=float)
        eligible_idx = np.where(eligible_mask)[0]
        coords[eligible_idx] = coords_eligible
    except Exception as exc:
        logger.warning("Skipping co-outlier UMAP after failure: %s", exc)
        return None, "failed"

    return (
        _projection_frame(
            sample_ids, subpopulations, "UMAP1", "UMAP2", coords[:, 0], coords[:, 1]
        ),
        "ok",
    )


def _compute_tsne_coordinates(
    similarity: pd.DataFrame,
    sample_ids: list,
    eligible_mask: np.ndarray,
    subpopulations: pd.DataFrame,
    random_state: int,
) -> tuple[Optional[pd.DataFrame], str]:
    n_samples = len(sample_ids)
    if n_samples < 4:
        logger.warning(
            "Skipping co-outlier t-SNE: need at least 4 samples (got %s)", n_samples
        )
        return None, "skipped_insufficient_samples"

    n_eligible = int(np.sum(eligible_mask))
    if n_eligible < 4:
        return (
            _nan_projection_frame(sample_ids, subpopulations, "TSNE1", "TSNE2"),
            "skipped_insufficient_eligible_samples",
        )

    eligible_ids = [sid for sid, ok in zip(sample_ids, eligible_mask) if ok]
    sim_eligible = similarity.loc[eligible_ids, eligible_ids].values.astype(float)
    distance = 1.0 - sim_eligible
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)

    perplexity = min(30, max(2, (n_eligible - 1) // 3))
    if perplexity >= n_eligible:
        perplexity = max(1, n_eligible - 1)

    try:
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=random_state,
        )
        coords_eligible = tsne.fit_transform(distance)
        coords = np.full((n_samples, 2), np.nan, dtype=float)
        eligible_idx = np.where(eligible_mask)[0]
        coords[eligible_idx] = coords_eligible
    except Exception as exc:
        logger.warning("Skipping co-outlier t-SNE after failure: %s", exc)
        return None, "failed"

    return (
        _projection_frame(
            sample_ids, subpopulations, "TSNE1", "TSNE2", coords[:, 0], coords[:, 1]
        ),
        "ok",
    )


@dataclass
class CoOutlierSimilarity:
    """Patient similarity and subpopulations from shared directional Z-score outliers."""

    similarity: pd.DataFrame
    subpopulations: Optional[pd.DataFrame] = None
    pca_coordinates: Optional[pd.DataFrame] = None
    umap_coordinates: Optional[pd.DataFrame] = None
    tsne_coordinates: Optional[pd.DataFrame] = None
    burden: Optional[pd.DataFrame] = None
    info: Optional[pd.DataFrame] = None

    def save(self, out_dir: str) -> dict[str, Path]:
        """Write co-outlier patient stratification outputs to out_dir."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        similarity_path = out_path / "cooutlier_patient_similarity.csv"
        self.similarity.to_csv(similarity_path, header=True, index=True)
        written["cooutlier_patient_similarity"] = similarity_path
        logger.info("Saved co-outlier patient similarity to %s", similarity_path)

        if self.subpopulations is not None:
            subpop_path = out_path / "cooutlier_patient_subpopulations.csv"
            self.subpopulations.to_csv(subpop_path, header=True, index=False)
            written["cooutlier_patient_subpopulations"] = subpop_path
            logger.info("Saved co-outlier subpopulations to %s", subpop_path)

        if self.burden is not None:
            burden_path = out_path / "cooutlier_patient_burden.csv"
            self.burden.to_csv(burden_path, header=True, index=False)
            written["cooutlier_patient_burden"] = burden_path
            logger.info("Saved co-outlier burden table to %s", burden_path)

        if self.pca_coordinates is not None:
            pca_path = out_path / "cooutlier_patient_pca.csv"
            self.pca_coordinates.to_csv(pca_path, header=True, index=False)
            written["cooutlier_patient_pca"] = pca_path
            logger.info("Saved co-outlier PCA coordinates to %s", pca_path)

        if self.umap_coordinates is not None:
            umap_path = out_path / "cooutlier_patient_umap.csv"
            self.umap_coordinates.to_csv(umap_path, header=True, index=False)
            written["cooutlier_patient_umap"] = umap_path
            logger.info("Saved co-outlier UMAP coordinates to %s", umap_path)

        if self.tsne_coordinates is not None:
            tsne_path = out_path / "cooutlier_patient_tsne.csv"
            self.tsne_coordinates.to_csv(tsne_path, header=True, index=False)
            written["cooutlier_patient_tsne"] = tsne_path
            logger.info("Saved co-outlier t-SNE coordinates to %s", tsne_path)

        if self.info is not None:
            info_path = out_path / "cooutlier_patient_info.csv"
            self.info.to_csv(info_path, header=True, index=False)
            written["cooutlier_patient_info"] = info_path
            logger.info("Saved co-outlier metadata to %s", info_path)

        return written


def compute_cooutlier_similarity(
    zscores: pd.DataFrame,
    z_threshold: float = 3.0,
    min_anomalies: int = 1,
    max_clusters: int = 10,
    min_samples_for_clustering: int = 4,
    random_state: int = 42,
) -> Optional[CoOutlierSimilarity]:
    """
    Stratify patients by shared extreme Z-score aberrations in the same direction.

    Args:
        zscores: Sample × protein Z-score matrix (in-memory ``Result.df_Z`` orientation).
        z_threshold: Absolute Z cutoff for up (>=) and down (<=) aberrations.
        min_anomalies: Minimum aberrant proteins required for clustering/projections.
        max_clusters: Maximum number of agglomerative clusters to consider.
        min_samples_for_clustering: Minimum eligible samples for automatic k selection.
        random_state: RNG seed for stochastic projections.

    Returns:
        CoOutlierSimilarity bundle, or None if fewer than two samples.
    """
    z = zscores.copy()
    if z.shape[0] < 2:
        logger.warning(
            "Skipping co-outlier similarity: fewer than two samples (%s)",
            z.shape[0],
        )
        return None

    similarity, burden, b_up, b_down = _compute_directional_jaccard(z, z_threshold)
    features = _sparse_cooutlier_features(b_up, b_down)
    sample_ids = list(similarity.index)
    eligible_mask = (
        burden.set_index("sampleID").loc[sample_ids, "n_total"].values >= min_anomalies
    )

    subpopulations, selected_k, selected_silhouette, cluster_status = (
        _cluster_cooutlier_subpopulations(
            similarity,
            burden,
            z_threshold=z_threshold,
            min_anomalies=min_anomalies,
            max_clusters=max_clusters,
            min_samples_for_clustering=min_samples_for_clustering,
        )
    )

    pca_coordinates, pca_status = _compute_pca_coordinates(
        features, sample_ids, eligible_mask, subpopulations, random_state
    )
    umap_coordinates, umap_status = _compute_umap_coordinates(
        similarity, sample_ids, eligible_mask, subpopulations, random_state
    )
    tsne_coordinates, tsne_status = _compute_tsne_coordinates(
        similarity, sample_ids, eligible_mask, subpopulations, random_state
    )

    n_with_anomaly = int((burden["n_total"] > 0).sum())
    n_eligible = int(np.sum(eligible_mask))

    info = pd.DataFrame(
        [
            {
                "similarity_method": SIMILARITY_METHOD,
                "z_threshold": z_threshold,
                "min_anomalies": min_anomalies,
                "n_samples": z.shape[0],
                "n_proteins": z.shape[1],
                "n_samples_with_any_anomaly": n_with_anomaly,
                "n_samples_eligible_for_clustering": n_eligible,
                "n_total_up_anomalies": int(burden["n_up"].sum()),
                "n_total_down_anomalies": int(burden["n_down"].sum()),
                "n_total_anomalies": int(burden["n_total"].sum()),
                "clustering_method": CLUSTERING_METHOD,
                "selected_k": selected_k,
                "selected_silhouette_score": selected_silhouette,
                "pca_method": PCA_METHOD if pca_coordinates is not None else np.nan,
                "pca_status": pca_status,
                "umap_method": UMAP_METHOD if umap_coordinates is not None else np.nan,
                "umap_status": umap_status,
                "tsne_method": TSNE_METHOD if tsne_coordinates is not None else np.nan,
                "tsne_status": tsne_status,
                "status": cluster_status,
            }
        ]
    )

    return CoOutlierSimilarity(
        similarity=similarity,
        subpopulations=subpopulations,
        pca_coordinates=pca_coordinates,
        umap_coordinates=umap_coordinates,
        tsne_coordinates=tsne_coordinates,
        burden=burden,
        info=info,
    )
