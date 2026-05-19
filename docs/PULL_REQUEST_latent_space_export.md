# Pull request: Latent-space export for PROTRIDER

## Summary

This change adds **latent-space extraction and CSV export** to the refactored `src/` layout, so users can inspect sample geometry in the learned encoder space **before residualization**. It also fixes a bug in PCA-based weight initialization when covariates are present.

No new CLI flags are required: latent files are written automatically when results are saved in **wide** format (including the default `protrider run --config config.yaml` workflow).

---

## Motivation

Residual outputs (`residuals.csv`, p-values, z-scores) reflect observed minus predicted protein intensities after the full model fit. That space can remove both technical and biological structure that is still present in the encoder latent representation.

Primary use case:

- Sample–sample similarity / clustering in latent space
- Checking whether latent dimensions align with known batch or biological covariates
- Comparing latent structure to residual-based outlier calls

Terminology used in code and docs:

| Concept | Shape | Source |
|--------|-------|--------|
| **Sample latent embeddings** | samples × q | Encoder output |
| **Protein loadings (SVD)** | proteins × q | `dataset.Vt` from SVD/OHT/PCA init |
| **Protein loadings (decoder)** | proteins × q | First `q` columns of linear decoder weights (`n_layers: 1` only) |

Proteins do not pass through the encoder; protein-side matrices are **loadings**, not embeddings.

---

## Changes

### 1. New module: `src/protrider/latent.py`

- **`LatentSpace`** dataclass:
  - `samples` — encoder latent matrix (samples × q)
  - `protein_loadings_svd` — optional SVD/OHT loadings (proteins × q)
  - `protein_loadings_decoder` — optional learned decoder loadings (proteins × q)
- **`extract_latent_space(dataset, model, q)`** — builds `LatentSpace` from a fitted model
- **`LatentSpace.save(out_dir)`** — writes CSVs and returns a dict of logical name → `Path`

Extraction details:

- **Samples:** `model.encoder(dataset.X, cond=dataset.covariates)` under `torch.no_grad()` (same path as `ProtriderAutoencoder.forward()` for the encoder; no manual `prot_means` subtraction).
- **SVD loadings:** `dataset.Vt[:q, :n_proteins].T` when `Vt` exists; otherwise warning and skip (no silent recompute).
- **Decoder loadings:** only if `n_layers == 1` and not `presence_absence`; uses `decoder.model.weight[:, :q]` so covariate columns are excluded when `n_cov > 0`.

### 2. Pipeline integration: `src/protrider/pipeline.py`

- `Result.latent_space: Optional[LatentSpace] = None`
- `run()` calls `extract_latent_space()` after statistics, passes result into `_format_results()`
- `Result.save(..., format="wide")` calls `latent_space.save(out_dir)` after existing wide outputs
- Long format (`protrider_summary.csv`) is unchanged — latent files are **not** written for `format="long"`

### 3. Public API: `src/protrider/__init__.py`

- Exports `LatentSpace` in `__all__`

```python
from protrider import ProtriderConfig, run, LatentSpace

result, model_info, fit_params, gs_result = run(config)
Z_samples = result.latent_space.samples
svd_loadings = result.latent_space.protein_loadings_svd
dec_loadings = result.latent_space.protein_loadings_decoder  # None for multilayer models
```

### 4. Bug fix: `initialize_wPCA()` in `src/protrider/model/model.py`

When `init_pca=True` and covariates are used, encoder/decoder weight layouts are:

| Layer | Shape | Latent/protein block | Covariate block |
|-------|-------|----------------------|-----------------|
| Encoder | `(q, n_proteins + n_cov)` | first `n_proteins` cols | **trailing** `n_cov` cols |
| Decoder | `(n_proteins, q + n_cov)` | first `q` cols | **trailing** `n_cov` cols |

Previously, covariate weights were taken from the **leading** columns (`[:, 0:n_cov]`), which overwrote the wrong part of the matrix. The fix preserves trailing covariate columns and assigns `Vt_q` / `Vt_q.T` to the correct blocks.

### 5. Documentation: `README.md`

- New output files in the wide-format table
- **Latent-space outputs** section (vs residuals, multilayer caveat, OHT+covariates note)
- Cosine-similarity example using `latent_samples.csv`
- Python API examples for `result.latent_space.*`
- **Install or update from this repository** (`uv sync`, editable install, `git+https` upgrade)

### 6. Tests: `tests/test_latent_space.py`

| Test | What it checks |
|------|----------------|
| OHT linear | All three latent CSVs + shapes and index alignment |
| Multilayer (`n_layers=2`) | `latent_samples.csv` + SVD; **no** decoder loadings file |
| OHT + covariates | No hard error; decoder has `q` columns (not `q + n_cov`) |
| Regression | Existing wide outputs still written |
| `initialize_wPCA` | Trailing covariate columns preserved after PCA init |

---

## New output files (wide format)

Written to `out_dir` when `result.save(out_dir, format="wide")` or after `protrider run`:

| File | Description | When omitted |
|------|-------------|--------------|
| `latent_samples.csv` | Sample encoder embeddings (rows = samples, cols = `latent_1` … `latent_q`) | Never (if `latent_space` is populated) |
| `latent_protein_loadings_svd.csv` | Protein loadings from SVD/OHT init | If `dataset.Vt` is missing |
| `latent_protein_loadings_decoder.csv` | Linear decoder loadings | `n_layers > 1`, `presence_absence`, or non-linear decoder |

CSV orientation matches other wide outputs: **rows = entities** (samples or proteins), **columns = latent dimensions**.

---

## Behavior unchanged

- CLI: `protrider run --config config.yaml` — no new flags
- OHT + covariates: still **warning only** in CLI (not a hard error)
- Statistical outputs: `residuals.csv`, `pvals.csv`, `zscores.csv`, etc. — same computation order; latent extraction runs **after** stats but does not alter them
- Long-format summary: unchanged

---

## How to test (reviewer / author checklist)

### 1. Unit tests

```bash
cd PROTRIDER
uv sync
uv run pytest tests/ -q
```

Expected: all tests pass (including `tests/test_latent_space.py`).

### 2. CLI smoke (sample data)

```bash
uv run protrider run --config config.yaml
```

Check `output/` (or your `out_dir`) contains:

- Existing: `processed_input.csv`, `output.csv`, `residuals.csv`, `pvals.csv`, `pvals_adj.csv`, `zscores.csv`, …
- **New:** `latent_samples.csv`, `latent_protein_loadings_svd.csv`, `latent_protein_loadings_decoder.csv` (for default linear config)

```bash
uv run protrider plot --config config.yaml all
```

Plots should still run (no dependency on latent files).

### 3. Python API

```python
import protrider

config = protrider.load_config("config.yaml")  # or ProtriderConfig(...)
result, model_info, fit_params, gs_result = protrider.run(config)

assert result.latent_space is not None
print(result.latent_space.samples.shape)  # (n_samples, q)

result.save(config.out_dir, format="wide")
```

### 4. Latent similarity (optional)

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

Z = pd.read_csv("output/latent_samples.csv", index_col=0)
sim = pd.DataFrame(cosine_similarity(Z), index=Z.index, columns=Z.index)
sim.to_csv("output/sample_latent_cosine_similarity.csv")
```

### 5. Covariates + OHT (optional)

Use a config with `sample_annotation` and `cov_used`. Expect CLI warning:

`OHT has not been evaluated with covariates yet`

Run should complete; `latent_samples.csv` and decoder loadings should have **`q`** latent columns, not `q + n_cov`.

### 6. Multilayer (optional)

Set `n_layers: 2` in config. Expect `latent_samples.csv` and SVD loadings, but **no** `latent_protein_loadings_decoder.csv`.

---

## Files changed

| File | Change |
|------|--------|
| `src/protrider/latent.py` | **Added** — `LatentSpace`, `extract_latent_space`, `save` |
| `src/protrider/pipeline.py` | `Result.latent_space`, extraction in `run()`, save in wide format |
| `src/protrider/model/model.py` | Fix `initialize_wPCA` covariate column slicing |
| `src/protrider/__init__.py` | Export `LatentSpace` |
| `tests/test_latent_space.py` | **Added** — latent export and PCA init tests |
| `README.md` | Output table, latent section, API, install/update |
| `uv.lock` | Lockfile refresh (if present in your environment) |
| `docs/PULL_REQUEST_latent_space_export.md` | This document |

---

## Suggested PR title

**Add latent-space export (sample embeddings and protein loadings) and fix PCA init with covariates**

## Suggested commit message

```
Add latent-space export and fix PCA init covariate slicing

Expose encoder sample embeddings and protein loadings as wide-format
CSVs (latent_samples, SVD/decoder loadings). Integrate into run() and
Result.save without new CLI flags. Fix initialize_wPCA to preserve
trailing covariate weight columns. Add tests and README documentation.
```

---

## Post-merge notes

- PyPI release is separate; README includes instructions to install from this repo via `pip install -e .` or `git+https://...`.
- For multilayer or presence/absence models, document that only sample latents (+ SVD when available) are exported.
