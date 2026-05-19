# Changes: latent-space export (feature branch)

This document describes what was added or fixed on the `feature/latent-space-export` branch, **why** each change was made, and **how** it works at a high level. For hands-on verification, see the [Test latent-space export locally](README.md#-test-latent-space-export-locally) section in `README.md`.

---

## Background

PROTRIDER models protein intensities with a conditional autoencoder:

```
samples × proteins  →  encoder  →  samples × q  →  decoder  →  samples × proteins
```

Outlier detection uses **residuals** (observed minus predicted) and derived p-values / z-scores. Residuals can remove both technical and biological structure that is still present in the **encoder latent space**.

**Goal of this work:** export the learned latent representation so users can explore sample–sample similarity, clustering, or alignment with covariates **before** interpreting residual-based outliers.

---

## Summary of changes

| Area | What changed |
|------|----------------|
| New module | `src/protrider/latent.py` — `LatentSpace`, `extract_latent_space()`, CSV export |
| Pipeline | `Result.latent_space`; extraction in `run()`; save with wide-format results |
| Bug fix | `initialize_wPCA()` — correct column slices when covariates are used |
| Tests | `tests/test_latent_space.py` |
| Docs | `README.md` — output table, latent-space section, local testing guide |
| Public API | `LatentSpace` exported from `protrider` |

No new CLI flags. `protrider run --config config.yaml` behaves as before, with additional CSV files in `out_dir` when results are saved in wide format.

---

## 1. New latent-space module

**File:** `src/protrider/latent.py`

### What

- **`LatentSpace`** — dataclass holding up to three `pandas.DataFrame` objects:
  - `samples` — sample latent embeddings (required)
  - `protein_loadings_svd` — optional SVD/OHT loadings
  - `protein_loadings_decoder` — optional linear decoder loadings
- **`extract_latent_space(dataset, model, q)`** — builds a `LatentSpace` from a fitted model and dataset
- **`LatentSpace.save(out_dir)`** — writes CSV files and returns a dict of logical names to `Path`

### Why

Keeps latent logic out of `pipeline.py`, makes unit testing straightforward, and gives API users a single object to inspect or save.

### How

**Sample embeddings (`latent_samples.csv`)**

- Under `torch.no_grad()`, run the same encoder path as the autoencoder forward pass:
  `model.encoder(dataset.X, cond=dataset.covariates)`.
- Rows = sample IDs (`dataset.data.index`), columns = `latent_1` … `latent_q`.
- Protein means are **not** subtracted manually; if the model was built with `prot_means`, `ConditionalEnDecoder` handles that internally.

**SVD protein loadings (`latent_protein_loadings_svd.csv`)**

- If `dataset.Vt` exists (from `perform_svd()` during OHT/PCA init), take the first `q` rows and protein columns:
  `Vt[:q, :n_proteins]`, then transpose to proteins × q.
- Rows = protein IDs (`dataset.data.columns`).
- If `Vt` is missing, log a warning and skip the file (no silent recompute).

**Decoder protein loadings (`latent_protein_loadings_decoder.csv`)**

- Only for **`n_layers == 1`** and **not** `presence_absence`.
- Read `model.decoder.model.weight` with shape `(n_proteins, q + n_cov)`.
- Use only the **first `q` columns** — the latent block. Trailing columns correspond to covariates when `n_cov > 0`.
- For multilayer or presence/absence models, log a warning and skip the file.

---

## 2. Pipeline integration

**File:** `src/protrider/pipeline.py`

### What

- `Result` has a new field: `latent_space: Optional[LatentSpace] = None`.
- After residuals, p-values, and z-scores are computed, `run()` calls `extract_latent_space(dataset, model, q)` and passes the result into `_format_results()`.
- `Result.save(..., format="wide")` calls `latent_space.save(out_dir)` after the existing wide CSV exports.
- Long format (`format="long"`) is unchanged — latent CSVs are **not** written there.

### Why

- Extraction must happen **after** the model is trained or loaded and **before** returning results, while `model` and `q` are still in scope.
- Hooking into wide `save()` matches the CLI (`cli.py` already calls `result.save(out_dir, format="wide")`) without new user-facing options.

### How

```text
run(config)
  → load/train model, inference, statistics (unchanged)
  → extract_latent_space(dataset, model, q)
  → Result(..., latent_space=...)
  → user or CLI: result.save(out_dir, format="wide")
       → existing CSVs + latent_*.csv
```

Statistical outputs (`df_res`, `df_pvals`, etc.) are computed the same way as before; latent extraction does not alter them.

---

## 3. New output files

Written to `out_dir` (e.g. `output/`) on wide save or default CLI run:

| File | Shape | When written |
|------|-------|----------------|
| `latent_samples.csv` | samples × q | Always (if `latent_space` is set) |
| `latent_protein_loadings_svd.csv` | proteins × q | When `dataset.Vt` is available |
| `latent_protein_loadings_decoder.csv` | proteins × q | Linear model only (`n_layers: 1`, not presence/absence) |

**Terminology**

- **Sample latent embeddings** — values in `latent_samples.csv` (encoder output).
- **Protein loadings** — weights relating proteins to latent dimensions (SVD or decoder), not “protein embeddings” (proteins do not pass through the encoder).

**vs `residuals.csv`**

- Residuals = observed − predicted intensities after the full model.
- Latent samples = encoder coordinates used to reconstruct intensities; useful when residual space is too denoised for sample-level structure.

---

## 4. Bug fix: PCA initialization with covariates

**File:** `src/protrider/model/model.py` — `ProtriderAutoencoder.initialize_wPCA()`

### What

Covariate weight columns are now taken from the **trailing** part of encoder/decoder weight matrices, not the leading columns.

### Why

With covariates, linear layer shapes are:

- Encoder: `(q, n_proteins + n_cov)` — proteins first, covariates last.
- Decoder: `(n_proteins, q + n_cov)` — latent dimensions first, covariates last.

The old code used `[:, 0:n_cov]`, which preserved the wrong columns and overwrote the protein/latent blocks when applying `Vt_q` from PCA.

### How

After building `Vt_q` with shape `(q, n_proteins)`:

```python
# Encoder: [Vt_q | covariate_weights]
cov_enc_init = enc_layer.weight.data[:, n_proteins:]
enc_layer.weight.data.copy_(torch.cat([Vt_q, cov_enc_init], dim=1))

# Decoder: [Vt_q.T | covariate_weights]
cov_dec_init = dec_layer.weight.data[:, n_latent:]
dec_layer.weight.data.copy_(torch.cat([Vt_q.T, cov_dec_init], dim=1))
```

Encoder bias is still set from protein means as before. Covered by `TestInitializeWPCA` in `tests/test_latent_space.py`.

---

## 5. Public Python API

**File:** `src/protrider/__init__.py`

### What

`LatentSpace` is exported in `__all__` alongside `ProtriderConfig`, `run`, `Result`, etc.

### Why

Users running `protrider.run(config)` can inspect latents in memory without re-reading CSVs.

### How

```python
import protrider

result, model_info, fit_params, gs_result = protrider.run(config)

Z = result.latent_space.samples
svd = result.latent_space.protein_loadings_svd
dec = result.latent_space.protein_loadings_decoder  # None if multilayer

result.save(config.out_dir, format="wide")  # also writes latent_*.csv
```

---

## 6. Tests

**File:** `tests/test_latent_space.py`

| Test | Purpose |
|------|---------|
| OHT linear pipeline | All three latent CSVs; shapes and index alignment |
| Multilayer (`n_layers=2`) | Sample + SVD files; no decoder loadings |
| OHT + covariates | Run completes; decoder has `q` columns, not `q + n_cov` |
| Regression | Existing wide outputs still created |
| `initialize_wPCA` | Trailing covariate columns preserved |

Run: `uv run pytest tests/ -q` or `uv run pytest tests/test_latent_space.py -v`.

---

## 7. Documentation

**File:** `README.md`

- Output file table extended with the three `latent_*.csv` files.
- Collapsible **Latent-space outputs** (definitions, vs residuals, caveats).
- **Test latent-space export locally** — checkout branch, install, pytest, CLI run, file checks.
- Python API snippet for `result.latent_space.*`.

---

## 8. Behaviour unchanged

| Topic | Behaviour |
|-------|-----------|
| CLI | No new flags; `protrider run --config config.yaml` unchanged |
| OHT + covariates | Warning only (not a hard error); latent export still runs |
| Long-format export | No latent files |
| Checkpointing | Unchanged |
| Outlier statistics | Same computation; latent step is additive |

---

## Files touched

| File | Role |
|------|------|
| `src/protrider/latent.py` | **New** — extraction and save |
| `src/protrider/pipeline.py` | Wire into `run()` and `Result.save(wide)` |
| `src/protrider/model/model.py` | PCA init covariate slice fix |
| `src/protrider/__init__.py` | Export `LatentSpace` |
| `tests/test_latent_space.py` | **New** — tests |
| `README.md` | User-facing docs and testing guide |
| `CHANGES.md` | This document |

---

## Branch and status

- **Branch:** `feature/latent-space-export`
- **Fork:** [github.com/mityasmirnov/PROTRIDER](https://github.com/mityasmirnov/PROTRIDER)
- **Not yet on `main`:** merge or test locally first; see README testing section.
