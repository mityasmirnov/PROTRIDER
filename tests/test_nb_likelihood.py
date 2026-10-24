"""likelihood='nb': raw counts, negative-binomial autoencoder, NB test on the fitted means (an OUTRIDER-style caller)."""
import numpy as np
import pandas as pd
import pytest

from protrider import ProtriderConfig, run
from protrider.stats import fit_nb_theta_ml, get_pvals_nb, init_nb_theta


def _counts(n=80, g=120, rank=3, theta=20.0, seed=0):
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 0.6, (n, rank)) @ rng.normal(0, 0.6, (rank, g))
    sf = np.exp(rng.normal(0, 0.2, n))
    mu = sf[:, None] * np.exp(rng.uniform(2, 6, g) + factors)
    return rng.negative_binomial(theta, theta / (theta + mu)).astype(float), mu


def _config(path, out, **kw):
    return ProtriderConfig(out_dir=str(out), input_intensities=str(path), index_col="geneID", likelihood="nb", find_q_method="3", n_epochs=400,
                           lr=1e-3, device="cpu", seed=1, export_latent_space=False, export_patient_similarity=False,
                           export_cooutlier_patient_similarity=False, n_jobs=1, **kw)


def test_nb_pvalues_are_calibrated_and_theta_is_recovered():
    counts, mu = _counts(n=300, g=60, theta=15.0)
    mask = np.zeros_like(counts, dtype=bool)
    theta = fit_nb_theta_ml(counts, mu, mask)
    assert 8 < np.median(theta) < 30  # grid ML around the true means recovers theta 15
    p = get_pvals_nb(counts, mu, theta, mask)
    assert abs(np.mean(p < 0.05) - 0.05) < 0.02  # two-sided NB p is calibrated under the model
    th0 = init_nb_theta(counts, mu, mask)
    assert np.all((th0 >= 0.1) & (th0 <= 1e4))


def test_nb_pipeline_ranks_a_planted_loss_first_and_returns_counts_outputs(tmp_path):
    counts, _ = _counts()
    gene = int(np.argmax(counts.mean(axis=0)))
    counts[7, gene] = 0.0  # planted loss of the most expressed gene in sample 7
    df = pd.DataFrame(counts.T, index=[f"g{i}" for i in range(counts.shape[1])], columns=[f"s{j}" for j in range(counts.shape[0])])
    df.index.name = "geneID"
    path = tmp_path / "counts.tsv"
    df.to_csv(path, sep="\t")
    result, info, fit_params, _ = run(_config(path, tmp_path / "out"))
    assert fit_params is None and result.pval_dist == "nb"
    assert result.df_pvals.shape == (80, 120) and result.df_expected_counts.shape == (80, 120)
    assert result.df_pvals.loc["s7"].idxmin() == f"g{gene}"
    assert result.df_Z.loc["s7", f"g{gene}"] < -3 and result.log2fc.loc["s7", f"g{gene}"] < -2
    assert (result.df_expected_counts > 0).all().all()
    assert np.isfinite(result.df_pvals.to_numpy()).all()
    result.save(str(tmp_path / "out"))
    assert (tmp_path / "out" / "expected_counts.csv").is_file()


def test_nb_config_validation():
    with pytest.raises(ValueError):
        ProtriderConfig(input_intensities="x.tsv", out_dir="o", likelihood="nb", find_q_method="gs")
    with pytest.raises(ValueError):
        ProtriderConfig(input_intensities="x.tsv", out_dir="o", likelihood="nb", presence_absence=True)
    assert ProtriderConfig(input_intensities="x.tsv", out_dir="o", likelihood="nb", find_q_method="OHT").likelihood == "nb"
    assert ProtriderConfig(input_intensities="x.tsv", out_dir="o").likelihood == "gaussian"  # default unchanged
