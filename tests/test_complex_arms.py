"""
Tests for Update 11g complex-contamination remediation arms (docs/19 §5.4).

All synthetic/numpy-level — no training run, no GPU. These check the library
functions behave correctly in isolation; they do not run the prevalence grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from protrider.complex_arms import (
    apply_group_masking,
    carrier_excluded_fit_residuals,
    covariate_only_complex_score,
    latent_complex_alignment_audit,
    leave_one_complex_out_score,
    perform_svd_group_masked,
    robust_reference_null,
)


def _synthetic_cohort(n_samples=40, n_proteins=20, seed=0):
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n_samples)]
    proteins = [f"P{i}" for i in range(n_proteins)]
    batch = rng.integers(0, 2, size=n_samples).astype(float)
    latent = rng.normal(size=n_samples)
    values = (
        0.5 * latent[:, None]
        + 0.3 * batch[:, None]
        + rng.normal(scale=0.3, size=(n_samples, n_proteins))
    )
    intensities = pd.DataFrame(values, index=samples, columns=proteins)
    covariates = pd.DataFrame({"batch": batch}, index=samples)
    return intensities, covariates, samples, proteins


class TestCovariateOnlyComplexScore:
    def test_runs_and_is_batch_explainable(self):
        intensities, covariates, samples, proteins = _synthetic_cohort()
        members = proteins[:4]
        z = covariate_only_complex_score(intensities, covariates, members)
        assert list(z.index) == samples
        assert z.notna().sum() > 0

    def test_missing_members_raises(self):
        intensities, covariates, _samples, _proteins = _synthetic_cohort()
        with pytest.raises(ValueError):
            covariate_only_complex_score(intensities, covariates, ["NOT_A_PROTEIN"])

    def test_disease_signal_outside_covariates_is_not_absorbed(self):
        """A complex-wide shift not explained by covariates should still show up."""
        intensities, covariates, samples, proteins = _synthetic_cohort()
        members = proteins[:4]
        affected = samples[:10]
        intensities.loc[affected, members] -= 3.0
        z = covariate_only_complex_score(intensities, covariates, members)
        assert z.loc[affected].mean() < z.loc[samples[10:]].mean()


class TestGroupMasking:
    def test_apply_group_masking_sets_mean_and_mask(self):
        n_s, n_p = 6, 5
        X = np.arange(n_s * n_p, dtype=float).reshape(n_s, n_p)
        mask = np.zeros((n_s, n_p), dtype=bool)
        prot_means = np.full(n_p, -1.0)
        X2, mask2 = apply_group_masking(X, mask, prot_means, member_col_idx=[1, 3], sample_idx=[0, 2])
        assert np.array_equal(X2[[0, 2]][:, [1, 3]], np.full((2, 2), -1.0))
        assert mask2[[0, 2]][:, [1, 3]].all()
        # original inputs untouched
        assert not mask.any()
        assert X[0, 1] != -1.0

    def test_group_masking_leaves_other_samples_and_columns_alone(self):
        n_s, n_p = 4, 4
        X = np.ones((n_s, n_p))
        mask = np.zeros((n_s, n_p), dtype=bool)
        X2, mask2 = apply_group_masking(X, mask, np.zeros(n_p), member_col_idx=[0], sample_idx=[1])
        assert not mask2[0].any()
        assert not mask2[2:].any()
        assert mask2[1, 0] and not mask2[1, 1:].any()

    def test_perform_svd_group_masked_zeroes_columns_before_svd(self):
        rng = np.random.default_rng(1)
        centered = rng.normal(size=(10, 6))
        _, _, vt_full = np.linalg.svd(centered, full_matrices=False)
        _, _, vt_masked = perform_svd_group_masked(centered, member_col_idx=[2])
        assert not np.allclose(vt_full, vt_masked)
        zeroed = np.array(centered, copy=True)
        zeroed[:, 2] = 0.0
        _, _, vt_expected = np.linalg.svd(zeroed, full_matrices=False)
        assert np.allclose(vt_masked, vt_expected)


class TestLeaveOneComplexOut:
    def test_runs_and_scores_members_vs_rest(self):
        intensities, covariates, samples, proteins = _synthetic_cohort()
        members = proteins[:3]
        z = leave_one_complex_out_score(intensities, covariates, members, n_comp=3)
        assert list(z.index) == samples
        assert z.notna().sum() > 0

    def test_no_non_members_raises(self):
        intensities, covariates, _samples, proteins = _synthetic_cohort(n_proteins=3)
        with pytest.raises(ValueError):
            leave_one_complex_out_score(intensities, covariates, proteins, n_comp=2)


class TestCarrierExcludedAndRobustNull:
    def _residuals(self, n_samples=30, n_proteins=8, seed=2):
        rng = np.random.default_rng(seed)
        samples = [f"S{i}" for i in range(n_samples)]
        proteins = [f"G{i}" for i in range(n_proteins)]
        data = rng.normal(size=(n_samples, n_proteins))
        return pd.DataFrame(data, index=samples, columns=proteins), samples

    def test_carrier_excluded_fit_drops_named_samples(self):
        res, samples = self._residuals()
        excluded = samples[:5]
        fit = carrier_excluded_fit_residuals(res, excluded, dis="gaussian")
        assert len(fit.genes) == res.shape[1]
        direct = carrier_excluded_fit_residuals(res.loc[samples[5:]].pipe(lambda d: d), [], dis="gaussian")
        assert np.allclose(fit.means, direct.means)

    def test_carrier_excluded_fit_raises_if_too_few_remain(self):
        res, samples = self._residuals(n_samples=3)
        with pytest.raises(ValueError):
            carrier_excluded_fit_residuals(res, samples[:2], dis="gaussian")

    def test_robust_null_matches_median_mad(self):
        res, _samples = self._residuals()
        fit = robust_reference_null(res)
        expected_mu = np.median(res.to_numpy(), axis=0)
        assert np.allclose(fit.means, expected_mu)
        assert fit.degrees_freedoms is None

    def test_robust_null_reference_only_differs_from_full(self):
        res, samples = self._residuals()
        res_copy = res.copy()
        # contaminate half the cohort with a shared shift, as a "high prevalence" stand-in
        res_copy.loc[samples[: len(samples) // 2]] -= 4.0
        reference = samples[len(samples) // 2 :]
        ref_only = robust_reference_null(res_copy, reference_sample_ids=reference)
        full = robust_reference_null(res_copy)
        assert not np.allclose(ref_only.means, full.means)
        assert np.allclose(ref_only.means, np.median(res_copy.loc[reference].to_numpy(), axis=0))

    def test_robust_null_raises_with_too_few_reference_rows(self):
        res, samples = self._residuals()
        with pytest.raises(ValueError):
            robust_reference_null(res, reference_sample_ids=samples[:1])


class TestLatentComplexAlignmentAudit:
    def test_flags_enriched_component(self):
        proteins = [f"P{i}" for i in range(10)]
        members = proteins[:3]
        loadings = pd.DataFrame(
            {
                "latent_1": [0.9, 0.8, 0.85] + [0.01] * 7,
                "latent_2": [0.1] * 10,
            },
            index=proteins,
        )
        audit = latent_complex_alignment_audit(loadings, {"COMPLEX_A": members})
        row1 = audit[(audit.complex == "COMPLEX_A") & (audit.component == "latent_1")].iloc[0]
        row2 = audit[(audit.complex == "COMPLEX_A") & (audit.component == "latent_2")].iloc[0]
        assert row1.enrichment_ratio > 5.0
        assert row2.enrichment_ratio == pytest.approx(1.0, abs=0.05)

    def test_skips_complex_with_too_few_members_present(self):
        proteins = [f"P{i}" for i in range(5)]
        loadings = pd.DataFrame({"latent_1": np.ones(5)}, index=proteins)
        audit = latent_complex_alignment_audit(loadings, {"TINY": ["P0", "NOT_PRESENT"]})
        assert audit.empty
