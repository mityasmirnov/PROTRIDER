"""
Update 11g — PROTRIDER complex-contamination remediation arms (docs/19 §5.4).

Five variants, each additive and opt-in. None of this is wired into the
default pipeline/CLI (`protrider.pipeline.run`, the `protrider` CLI), so the
baseline fit/score path behaves exactly as before — this module is the flag:
nothing here runs unless a caller imports and calls it explicitly. Building
this is not running the prevalence-grid experiments in docs/19 §5.4 (vary
prevalence 0.5-50%, cohort size separately); those still need separate
confirmation before any real or synthetic-control run.

1. covariate_only_complex_score     — complex z-score from covariates only;
   nothing about an AE/disease programme can enter this score.
2. apply_group_masking /
   perform_svd_group_masked         — hide a complex's intensities and
   detection flags the same way PROTRIDER already represents a real missing
   value (X <- protein mean, mask <- True) — "neural LOCO" group masking,
   for the loss and for PCA init.
3. leave_one_complex_out_score      — reduced-rank (PCA) prediction of a
   complex's eigengene from the rest of the proteome, the complex analogue
   of Scripts/ml_harness/cross_modal.py's Arm C.
4. carrier_excluded_fit_residuals   — stats.fit_residuals() restricted to a
   reference subset (an EVAdb carrier-exclusion list, once available).
5. robust_reference_null           — median/MAD null, optionally
   reference-only ("do not fit the null on all residuals").

Plus latent_complex_alignment_audit: do a fitted model's latent factors line
up with CORUM/MitoCarta membership (docs/19 §5.4 latent audit).

Pure numpy/pandas/scipy — no torch, no GPU — so all of this is unit-tested
on synthetic data without a training run (tests/test_complex_arms.py).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stats import FitParameters
from .stats import fit_residuals as _fit_residuals_baseline

__all__ = [
    "covariate_only_complex_score",
    "apply_group_masking",
    "perform_svd_group_masked",
    "leave_one_complex_out_score",
    "carrier_excluded_fit_residuals",
    "robust_reference_null",
    "latent_complex_alignment_audit",
]


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge; does not penalize the intercept (col 0)."""
    xtx = X.T @ X
    pen = np.eye(xtx.shape[0]) * lam
    pen[0, 0] = 0.0
    try:
        return np.linalg.solve(xtx + pen, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def covariate_only_complex_score(
    intensities: pd.DataFrame,
    covariates: pd.DataFrame,
    complex_members: Sequence[str],
    *,
    lam: float = 1.0,
) -> pd.Series:
    """
    Per-sample complex z-score explainable by covariates alone (docs/19 §5.4
    "covariate-only complex score"): regress each member protein on
    covariates only (plex/batch/channel/sex/QC — no AE, no other proteins),
    then combine the per-member z-scores Stouffer-style into one complex
    score. Nothing about a disease programme can enter this score by
    construction — it is the most orthogonal check in the table, at the cost
    of missing anything not explained by the listed covariates.

    intensities: samples x proteins, log-intensity (NaN allowed).
    covariates: samples x covariates, no intercept column needed.
    """
    members = [m for m in complex_members if m in intensities.columns]
    if not members:
        raise ValueError("no complex members found in intensities columns")
    X = np.hstack([np.ones((covariates.shape[0], 1)), covariates.to_numpy(dtype=float)])
    member_z = pd.DataFrame(index=intensities.index, columns=members, dtype=float)
    for m in members:
        y = intensities[m].to_numpy(dtype=float)
        obs = np.isfinite(y)
        if obs.sum() < 5:
            continue
        beta = _ridge_fit(X[obs], y[obs], lam=lam)
        resid = y - X @ beta
        finite = resid[obs]
        scale = float(np.std(finite, ddof=1)) if finite.size > 1 else 1e-6
        scale = scale if np.isfinite(scale) and scale >= 1e-6 else 1e-6
        member_z[m] = np.where(obs, resid / scale, np.nan)
    n_avail = member_z.notna().sum(axis=1)
    with np.errstate(invalid="ignore"):
        complex_z = member_z.sum(axis=1, skipna=True) / np.sqrt(n_avail.where(n_avail > 0))
    complex_z.name = "covariate_only_complex_z"
    return complex_z


def apply_group_masking(
    X: np.ndarray,
    mask: np.ndarray,
    prot_means: np.ndarray,
    member_col_idx: Sequence[int],
    sample_idx: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    "Neural LOCO" group masking (docs/19 §5.4 "group-masked PROTRIDER"): hide
    a complex's intensities AND detection flags for the selected samples the
    same way ProtriderDataset already represents a real missing value
    (X <- protein mean, mask <- True), so the loss excludes those entries and
    the AE never sees them for those samples.

    X, mask: (n_samples, n_proteins) — pass copies of dataset.X.numpy() /
    dataset.mask if calling from a torch context; this function never
    mutates its inputs.
    """
    X2 = np.array(X, copy=True)
    mask2 = np.array(mask, copy=True)
    means = np.asarray(prot_means).reshape(-1)
    rows = np.arange(X.shape[0]) if sample_idx is None else np.asarray(list(sample_idx))
    for j in member_col_idx:
        X2[np.ix_(rows, [j])] = means[j]
        mask2[np.ix_(rows, [j])] = True
    return X2, mask2


def perform_svd_group_masked(
    centered_log_data_noNA: np.ndarray,
    member_col_idx: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-init hook mirroring PCADataset.perform_svd(), but with a complex's
    columns zeroed first (docs/19 §5.4 "masked ... in PCA init"). Does not
    mutate the dataset; a caller wiring this into init_wPCA would pass the
    returned Vt instead of dataset.Vt for that one run.
    """
    X = np.array(centered_log_data_noNA, copy=True)
    for j in member_col_idx:
        X[:, j] = 0.0
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U, s, Vt


def leave_one_complex_out_score(
    intensities: pd.DataFrame,
    covariates: pd.DataFrame,
    complex_members: Sequence[str],
    *,
    n_comp: int = 5,
    lam: float = 1.0,
) -> pd.Series:
    """
    Reduced-rank (PCA) prediction of a complex's eigengene (mean member
    log-intensity) from the rest of the proteome + covariates — the complex
    analogue of Scripts/ml_harness/cross_modal.py's Arm C (docs/19 §5.4 LOCO
    row). This fits once on the given reference (a diagnostic/reference-level
    score, not a per-sample LOO ranking); callers wanting a LOO version
    should loop held-out samples the way cross_modal.loo_z_scores does.
    """
    members = [m for m in complex_members if m in intensities.columns]
    if not members:
        raise ValueError("no complex members found in intensities columns")
    others = [c for c in intensities.columns if c not in members]
    if not others:
        raise ValueError("no non-member proteins to predict the complex from")
    eigengene = intensities[members].mean(axis=1, skipna=True)
    rest = intensities[others].to_numpy(dtype=float)
    mean = np.nanmean(rest, axis=0)
    Xc = np.nan_to_num(rest - mean, nan=0.0)
    if Xc.shape[0] < 2 or Xc.shape[1] < 1:
        return pd.Series(np.nan, index=intensities.index, name="loco_complex_z")
    _, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_comp, vt.shape[0], max(0, Xc.shape[0] - 1))
    scores = Xc @ vt[:k].T if k > 0 else np.zeros((Xc.shape[0], 0))
    X = np.hstack([np.ones((intensities.shape[0], 1)), covariates.to_numpy(dtype=float), scores])
    y = eigengene.to_numpy(dtype=float)
    obs = np.isfinite(y)
    if obs.sum() < 5:
        return pd.Series(np.nan, index=intensities.index, name="loco_complex_z")
    beta = _ridge_fit(X[obs], y[obs], lam=lam)
    resid = y - X @ beta
    finite = resid[obs]
    mad = float(np.median(np.abs(finite - np.median(finite)))) if finite.size > 1 else 0.0
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(finite, ddof=1)) if finite.size > 1 else 1e-6
    scale = scale if np.isfinite(scale) and scale >= 1e-6 else 1e-6
    z = np.where(obs, resid / scale, np.nan)
    return pd.Series(z, index=intensities.index, name="loco_complex_z")


def carrier_excluded_fit_residuals(
    residuals: pd.DataFrame,
    excluded_sample_ids: Iterable[str],
    **fit_kwargs,
) -> FitParameters:
    """
    stats.fit_residuals() restricted to non-carrier rows (docs/19 §5.4
    "genotype-/label-aware reference"). The carrier-exclusion list itself
    (solved-in-G, rare protein-altering/biallelic in G or assembly factors,
    + relatives) must come from a frozen EVAdb snapshot — this function does
    not invent one; callers pass it in explicitly.
    """
    excluded = set(excluded_sample_ids)
    kept = [s for s in residuals.index if s not in excluded]
    if len(kept) < 2:
        raise ValueError("fewer than 2 reference samples remain after carrier exclusion")
    return _fit_residuals_baseline(residuals.loc[kept], **fit_kwargs)


def robust_reference_null(
    residuals: pd.DataFrame,
    reference_sample_ids: Iterable[str] | None = None,
) -> FitParameters:
    """
    Median/MAD null fit (docs/19 §5.4: "do not fit the null on all
    residuals"). If reference_sample_ids is given, the null is fit on that
    reference subset only, then can be applied to the full cohort via
    stats.get_pvals(..., dis='gaussian') using these mu/sigma — median/MAD is
    the robustness here, not a new distributional family.
    """
    if reference_sample_ids is not None:
        ref = set(reference_sample_ids)
        rows = [s for s in residuals.index if s in ref]
        if len(rows) < 2:
            raise ValueError("fewer than 2 reference samples for robust null")
        fit_frame = residuals.loc[rows]
    else:
        fit_frame = residuals
    matrix = fit_frame.to_numpy(dtype=float)
    mu = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - mu), axis=0)
    sigma = 1.4826 * mad
    finite_positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    fallback = float(np.median(finite_positive)) * 1e-3 if finite_positive.size else 1e-6
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, max(fallback, 1e-9))
    return FitParameters(
        genes=np.array(residuals.columns), sigmas=sigma, means=mu, degrees_freedoms=None
    )


def latent_complex_alignment_audit(
    protein_loadings: pd.DataFrame,
    complex_membership: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """
    Does any latent factor line up with a CORUM/MitoCarta complex (docs/19
    §5.4 latent audit)? For each (complex, component), report the ratio of
    mean |loading| on members vs non-members — appreciably >1 flags a disease
    axis to drop or penalise before scoring that complex. Diagnostic only,
    not a fix by itself (docs/19 table).

    protein_loadings: proteins x latent components, e.g.
    LatentSpace.protein_loadings_svd or .protein_loadings_decoder.
    complex_membership: complex name -> member protein ids.
    """
    rows = []
    all_proteins = set(protein_loadings.index)
    for complex_name, members in complex_membership.items():
        member_idx = [m for m in members if m in all_proteins]
        if len(member_idx) < 2:
            continue
        non_member_idx = [p for p in protein_loadings.index if p not in member_idx]
        if not non_member_idx:
            continue
        for comp in protein_loadings.columns:
            member_abs = protein_loadings.loc[member_idx, comp].abs().mean()
            other_abs = protein_loadings.loc[non_member_idx, comp].abs().mean()
            ratio = float(member_abs / other_abs) if other_abs > 0 else float("nan")
            rows.append(
                {
                    "complex": complex_name,
                    "component": comp,
                    "enrichment_ratio": ratio,
                    "n_members": len(member_idx),
                }
            )
    return pd.DataFrame(rows)
