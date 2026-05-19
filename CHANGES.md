# Changes

## Unreleased

### Added

- Co-outlier patient stratification based on directional Jaccard overlap of extreme Z-scores.
- Module `src/protrider/cooutlier_similarity.py`.
- New outputs: `cooutlier_patient_similarity.csv`, `cooutlier_patient_subpopulations.csv`, `cooutlier_patient_burden.csv`, `cooutlier_patient_pca.csv`, `cooutlier_patient_umap.csv`, `cooutlier_patient_tsne.csv`, `cooutlier_patient_info.csv`.
- Plot functions and CLI commands: `cooutlier_patient_similarity`, `cooutlier_patient_pca`, `cooutlier_patient_umap`, `cooutlier_patient_tsne` (included in `plot all`, skipped with a warning if CSVs are missing).
- Config: `export_cooutlier_patient_similarity`, `z_threshold`, `cooutlier_min_anomalies`, `cooutlier_max_clusters`, `cooutlier_min_samples_for_clustering`.
- Method: sparse up/down binary matrices at `z_threshold`, same-direction intersections via sparse multiplication, directional Jaccard similarity, average-linkage clustering on `1 - Jaccard` with silhouette-guided *k*, TruncatedSVD / UMAP / t-SNE projections for visualization.
- Does not alter p-values, Z-scores, residuals, latent exports, or latent-based patient similarity outputs.

- Patient similarity and subpopulation export from `latent_samples.csv`: RBF similarity matrix, Ward agglomerative clustering with silhouette-guided *k*, PCA coordinates, and metadata (`patient_similarity.csv`, `patient_subpopulations.csv`, `patient_latent_pca.csv`, `patient_similarity_info.csv`).
- Module `src/protrider/patient_similarity.py`; `Result.patient_similarity` populated after latent extraction when latents are available.
- Plots: `plot_patient_similarity`, `plot_patient_latent_pca`; CLI subcommands `patient_similarity` and `patient_latent_pca` (included in `plot all`, skipped with a warning if CSVs are missing).
- UMAP and t-SNE latent-space patient projection outputs: `patient_latent_umap.csv`, `patient_latent_tsne.csv` (visualization only; failures log warnings and skip the corresponding file without breaking the pipeline).
- Plot functions and CLI commands: `plot_patient_latent_umap`, `plot_patient_latent_tsne`, `protrider plot --config config.yaml patient_latent_umap`, `protrider plot --config config.yaml patient_latent_tsne`.
- Dependency: `umap-learn` (import `from umap import UMAP`).

### Added (cohort stability)

- Optional cohort stability analysis (subsampling / delete-d cohort perturbation, not classical bootstrap) for OHT runs.
- `protrider_summary_bs.csv` aggregated stability summary.
- Config: `cohort_stability`, `cohort_stability_n_runs`, `cohort_stability_min_runs`, `cohort_stability_max_runtime_min`, `cohort_stability_drop_fraction`, `cohort_stability_min_samples`, `cohort_stability_seed`, `cohort_stability_require_oht`, `cohort_stability_save_iteration_files`.
- Module `src/protrider/stability.py`.

### Fixed

- Co-outlier `_labels_to_subpopulation_names` now assigns one subpopulation label per sample (not one per cluster), so `cooutlier_patient_subpopulations.csv` and embedding plots no longer leave eligible samples with empty labels.

### Changed

- `Result.to_long_df()` for shared long-format construction; `Result.save(format="long")` uses it.

### Fixed (cohort stability review)

- Stability baseline always uses `to_long_df(include_all=True)` so full-cohort metrics are not dropped when `report_all: false`.
- Baseline-only rows after outer merge get valid `BS_*` counters and call rates.
- Temporary intensity subsets are always written as TSV (fixes parquet inputs).
- Annotation subset files use the annotation file’s delimiter, not the intensity file’s.
- `BS_N_RUNS_COMPLETED` added; missing/fraction denominators use completed runs, not requested.
- When `cohort_stability_save_iteration_files` is true, each iteration saves wide and long outputs.
- Stability iterations skip latent space, patient similarity, and co-outlier export for speed.

### Notes

- Preprocessing reruns each iteration; protein sets may differ (`BS_N_OBSERVED`).
- `PROTEIN_outlier_call_rate` uses observed iterations as denominator; `PROTEIN_outlier_call_rate_all_runs` uses completed runs.
- Stability iterations use isolated temp directories and do not load full-cohort `model.pt`.

---

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

---

# Detailed documentation: sample-level analyses (main)

The sections below document analyses added on `main` beyond the latent-space export notes above. They mirror the level of detail in **Changes: latent-space export** and are intended as a single reference for outputs, configuration, and interpretation.

---

## Cohort stability analysis (subsampling)

**Module:** `src/protrider/stability.py`  
**CLI:** enabled with `cohort_stability: true` in `config.yaml`; runs automatically after the standard full-cohort `protrider run` completes.

### What

Optional **cohort stability analysis** measures how sensitive outlier calls and protein-level effect sizes are to cohort composition. This is **not classical bootstrap** (no sampling with replacement). Each iteration:

1. Randomly **removes** a fraction of samples (`cohort_stability_drop_fraction`).
2. Writes temporary subset intensity (and optional annotation) files.
3. Reruns the full **OHT** pipeline **including preprocessing** in an isolated directory.
4. Uses a **fresh** `model.pt` per iteration (never loads the full-cohort checkpoint).

Results are aggregated into **`protrider_summary_bs.csv`** in `out_dir`.

### Requirements and skips

- `find_q_method: "OHT"` when `cohort_stability_require_oht: true` (default).
- Cohort must have **more than 30 samples** (hard skip with warning).
- Retained samples per iteration must be ≥ `cohort_stability_min_samples` (default 30).
- Optional `cohort_stability_max_runtime_min` stops after `cohort_stability_min_runs` iterations once the budget is exceeded.

### Configuration

| Parameter | Role |
|-----------|------|
| `cohort_stability` | Master switch (default: `false`) |
| `cohort_stability_n_runs` | Requested number of subsampling iterations (default: 100) |
| `cohort_stability_min_runs` | Minimum iterations before runtime budget may stop (default: 30) |
| `cohort_stability_max_runtime_min` | Optional wall-clock budget in minutes (`null` = run all requested) |
| `cohort_stability_drop_fraction` | Fraction of samples removed per iteration (default: 0.1) |
| `cohort_stability_min_samples` | Minimum retained samples per iteration (default: 30) |
| `cohort_stability_seed` | Seed for subsampling and per-iteration models (defaults to `seed`) |
| `cohort_stability_require_oht` | Enforce OHT when stability is enabled (default: `true`) |
| `cohort_stability_save_iteration_files` | Save wide + long iteration outputs under each temp dir (default: `false`) |

### Output: `protrider_summary_bs.csv`

One row per **sample–protein** pair (outer union of full cohort and all iterations). Key columns:

| Column | Meaning |
|--------|---------|
| `in_full_run` | Pair present in the full-cohort baseline |
| `PROTEIN_*_full` | Full-cohort metrics (from baseline; always all pairs, independent of `report_all`) |
| `BS_N_RUNS_REQUESTED` | Configured iteration count (metadata) |
| `BS_N_RUNS_COMPLETED` | Iterations actually executed (e.g. after runtime budget) |
| `BS_N_OBSERVED` | Iterations where the pair survived subsetting + preprocessing |
| `BS_N_MISSING` | `BS_N_RUNS_COMPLETED − BS_N_OBSERVED` |
| `BS_OBSERVED_FRACTION` | `BS_N_OBSERVED / BS_N_RUNS_COMPLETED` |
| `PROTEIN_outlier_call_count` | Iterations calling the pair an outlier |
| `PROTEIN_outlier_call_rate` | `call_count / BS_N_OBSERVED` (NaN if never observed) |
| `PROTEIN_outlier_call_rate_all_runs` | `call_count / BS_N_RUNS_COMPLETED` |
| `PROTEIN_FC_median`, `PROTEIN_FC_q025`, `PROTEIN_FC_q975`, … | Stability of effect sizes across observed iterations |

**Interpretation:** High `PROTEIN_outlier_call_rate` with stable FC quantiles suggests a robust outlier; low call rate or wide FC intervals suggests cohort-composition sensitivity.

### Integration details

- **Baseline for stability:** `cli.py` passes `result.to_long_df(include_all=True)` into `run_cohort_stability()`. Standard `protrider_summary.csv` still respects `report_all`.
- **Temporary I/O:** Intensity subsets are always written as `intensities_subset.tsv` (supports parquet inputs on the full run). Annotation subsets use the annotation file’s delimiter (`.tsv` vs `.csv`).
- **Performance:** Stability iterations set `export_latent_space=False`, `export_patient_similarity=False`, and `export_cooutlier_patient_similarity=False`.
- **Tests:** `tests/test_stability.py` (config, aggregation, parquet/mixed-delimiter subsets, runtime denominators, E2E smoke).

### Files

| File | Role |
|------|------|
| `src/protrider/stability.py` | Subsampling plan, subset I/O, iteration loop, aggregation |
| `src/protrider/cli.py` | Writes `protrider_summary_bs.csv` after main run |
| `src/protrider/pipeline.py` | `Result.to_long_df()` shared with long save |
| `tests/test_stability.py` | Unit and integration tests |

---

## Latent-based patient similarity and subpopulations

**Module:** `src/protrider/patient_similarity.py`  
**Trigger:** After latent extraction in `pipeline.run()` when `export_patient_similarity: true` (default) and `latent_space` is available.

### What

Sample-level **patient similarity** and **subpopulation** labels derived from encoder latent coordinates (`latent_samples.csv` in memory as `Result.latent_space.samples`):

- RBF kernel similarity matrix on latent vectors.
- Ward agglomerative clustering with silhouette-guided choice of *k* (within configured bounds).
- PCA, UMAP, and t-SNE coordinates for visualization (UMAP/t-SNE failures log warnings and skip files without failing the pipeline).

### Configuration

| Parameter | Role |
|-----------|------|
| `export_latent_space` | Extract and save latent CSVs (default: `true`; disabled in stability iterations) |
| `export_patient_similarity` | Compute patient similarity block (default: `true`; disabled in stability iterations) |

### Outputs (wide save / default CLI)

| File | Content |
|------|---------|
| `patient_similarity.csv` | Sample × sample RBF similarity |
| `patient_subpopulations.csv` | Sample ID → subpopulation label |
| `patient_latent_pca.csv` | PCA coordinates |
| `patient_latent_umap.csv` | UMAP coordinates (optional) |
| `patient_latent_tsne.csv` | t-SNE coordinates (optional) |
| `patient_similarity_info.csv` | Clustering metadata (*k*, method, etc.) |

### Plots and CLI

- `protrider plot --config config.yaml patient_similarity`
- `protrider plot --config config.yaml patient_latent_pca`
- `protrider plot --config config.yaml patient_latent_umap`
- `protrider plot --config config.yaml patient_latent_tsne`
- Included in `plot all` (skipped with warning if CSVs missing).

### Notes

- Does **not** change p-values, Z-scores, residuals, or outlier calls.
- Complements (does not replace) residual-based outlier interpretation.
- **Dependency:** `umap-learn` for UMAP (`from umap import UMAP`).

---

## Co-outlier patient stratification (directional Jaccard)

**Module:** `src/protrider/cooutlier_similarity.py`  
**Trigger:** After statistics in `pipeline.run()` when `export_cooutlier_patient_similarity: true` (default); uses Z-scores and `z_threshold`, not latents.

### What

**Co-outlier patient stratification** groups samples by overlap of extreme protein hits in the **same direction** (up vs down at `|Z| ≥ z_threshold`):

1. Build sparse up/down binary anomaly matrices per sample.
2. Same-direction co-anomalies via sparse matrix multiplication.
3. **Directional Jaccard** similarity between samples.
4. Average-linkage clustering on distance `1 − Jaccard`, silhouette-guided *k*.
5. TruncatedSVD / UMAP / t-SNE for visualization.

### Configuration

| Parameter | Role |
|-----------|------|
| `export_cooutlier_patient_similarity` | Master switch (default: `true`; disabled in stability iterations) |
| `z_threshold` | \|Z\| cutoff for defining anomalies |
| `cooutlier_min_anomalies` | Minimum anomalies per sample to include |
| `cooutlier_max_clusters` | Upper bound on *k* for clustering |
| `cooutlier_min_samples_for_clustering` | Minimum samples required to cluster |

### Outputs

| File | Content |
|------|---------|
| `cooutlier_patient_similarity.csv` | Directional Jaccard similarity matrix |
| `cooutlier_patient_subpopulations.csv` | Sample → co-outlier subpopulation |
| `cooutlier_patient_burden.csv` | Per-sample anomaly counts (up/down) |
| `cooutlier_patient_pca.csv` | TruncatedSVD projection |
| `cooutlier_patient_umap.csv` | UMAP projection (optional) |
| `cooutlier_patient_tsne.csv` | t-SNE projection (optional) |
| `cooutlier_patient_info.csv` | Method metadata |

### Plots and CLI

- `cooutlier_patient_similarity`, `cooutlier_patient_pca`, `cooutlier_patient_umap`, `cooutlier_patient_tsne`
- Included in `plot all` (skipped with warning if CSVs missing).

### Notes

- Does **not** alter p-values, Z-scores, residuals, latent exports, or latent-based patient similarity.
- Interprets **which samples share extreme hit patterns**, not encoder geometry.

---

## How the analyses relate

```text
protrider run (full cohort)
  → standard wide/long outputs (outliers, residuals, …)
  → latent_samples.csv (+ loadings)          [export_latent_space]
  → patient_similarity*.csv                  [export_patient_similarity]
  → cooutlier_patient*.csv                   [export_cooutlier_patient_similarity]
  → protrider_summary_bs.csv (optional)      [cohort_stability]
       each iteration: subset → OHT rerun → aggregate (heavy exports off)
```

| Analysis | Input space | Question answered |
|----------|-------------|-------------------|
| Latent export | Encoder latents | What low-dimensional structure did the model learn? |
| Patient similarity | Latent RBF + clustering | Which samples are close in latent space? |
| Co-outlier similarity | Directional Z-hit overlap | Which samples share extreme co-regulation patterns? |
| Cohort stability | Repeated subsampled OHT runs | How stable are calls and FCs if the cohort changes? |

---

## Tests (main analyses)

| Module | Test file |
|--------|-----------|
| Latent space | `tests/test_latent_space.py` |
| Patient similarity | `tests/test_patient_similarity.py`, `tests/test_patient_similarity_plots.py` |
| Co-outlier | `tests/test_cooutlier_similarity.py`, `tests/test_cooutlier_plots.py` |
| Cohort stability | `tests/test_stability.py` |

Run: `uv run pytest tests/ -q`

---

## Appendix: defaults, Python API, and plot artifacts

The following supplements the sections above with default configuration values, programmatic access, and saved figure paths. Nothing here replaces earlier release notes; it only adds detail.

### Default configuration values

| Parameter | Default | Applies to |
|-----------|---------|------------|
| `export_latent_space` | `true` | Latent CSV export |
| `export_patient_similarity` | `true` | Latent-based patient similarity |
| `export_cooutlier_patient_similarity` | `true` | Co-outlier stratification |
| `z_threshold` | `3.0` | Co-outlier up/down bins (\|Z\| cutoff; not `outlier_threshold`) |
| `cooutlier_min_anomalies` | `1` | Minimum aberrant proteins for clustering/projections |
| `cooutlier_max_clusters` | `10` | Max *k* for co-outlier agglomerative search |
| `cooutlier_min_samples_for_clustering` | `4` | Min eligible samples for automatic *k* |
| `cohort_stability` | `false` | Subsampling stability (off unless enabled) |

Validation in `ProtriderConfig.__post_init__`: `z_threshold > 0`, `cooutlier_min_anomalies >= 0`, `cooutlier_max_clusters >= 2`, `cooutlier_min_samples_for_clustering >= 2`.

### Python API (`protrider` package)

```python
import protrider

result, model_info, fit_params, gs_result = protrider.run(config)

# Latent space (when export_latent_space is true)
result.latent_space.samples
result.latent_space.protein_loadings_svd

# Latent-based patient similarity (when latents available)
result.patient_similarity.similarity
result.patient_similarity.subpopulations
result.patient_similarity.pca_coordinates

# Co-outlier stratification (when export_cooutlier_patient_similarity is true)
result.cooutlier_similarity.similarity
result.cooutlier_similarity.subpopulations
result.cooutlier_similarity.burden
result.cooutlier_similarity.pca_coordinates

# Standalone co-outlier computation from in-memory Z (samples × proteins)
from protrider import compute_cooutlier_similarity

co = compute_cooutlier_similarity(result.df_Z, z_threshold=3.0)
```

Public exports from `protrider`: `LatentSpace`, `PatientSimilarity`, `CoOutlierSimilarity`, `compute_patient_similarity`, `compute_cooutlier_similarity`.

### Co-outlier method (implementation reference)

- **Input:** `Result.df_Z` (samples × proteins). Do not use on-disk `zscores.csv` inside the pipeline (wide file is proteins × samples).
- **Binarization:** `Z >= z_threshold` → up; `Z <= -z_threshold` → down; NaN and non-extreme values contribute neither.
- **Intersection:** `C_ij = |up_i ∩ up_j| + |down_i ∩ down_j|` (sparse multiply). Opposite directions on the same protein do not count.
- **Similarity:** `C_ij / (burden_i + burden_j - C_ij)` when the union is positive; symmetric, values in [0, 1].
- **Diagonal:** `1` if the sample has ≥1 aberration, else `0` (zero-burden samples are not treated as maximally self-similar).
- **Clustering:** Distance `D = 1 - similarity`; **average** linkage on precomputed `D` (Ward is not used — it requires Euclidean feature vectors). *k* chosen by maximum silhouette among `2 … min(cooutlier_max_clusters, n_eligible - 1)`.
- **Low burden:** `n_total < cooutlier_min_anomalies` → subpopulation `unclassified_low_burden`, excluded from automatic *k* selection but retained in all tables.
- **Projections:** TruncatedSVD on sparse `[B_up | B_down]`; UMAP and t-SNE on precomputed Jaccard distance for eligible samples; ineligible samples appear with NaN coordinates. Failures log warnings only.

### Plot artifacts (`<out_dir>/plots/`)

| Analysis | PNG (under `plots/`) | CLI subcommand |
|----------|----------------------|----------------|
| Latent patient similarity | `patient_similarity_heatmap.png` | `patient_similarity` |
| Latent PCA | `patient_latent_pca.png` | `patient_latent_pca` |
| Latent UMAP | `patient_latent_umap.png` | `patient_latent_umap` |
| Latent t-SNE | `patient_latent_tsne.png` | `patient_latent_tsne` |
| Co-outlier similarity | `cooutlier_patient_similarity_heatmap.png` | `cooutlier_patient_similarity` |
| Co-outlier PCA-like | `cooutlier_patient_pca.png` | `cooutlier_patient_pca` |
| Co-outlier UMAP | `cooutlier_patient_umap.png` | `cooutlier_patient_umap` |
| Co-outlier t-SNE | `cooutlier_patient_tsne.png` | `cooutlier_patient_tsne` |

All plot commands are included in `protrider plot --config config.yaml all`. Missing CSVs produce a log warning and return `None` without failing the command.

### Files touched (co-outlier feature)

| Path | Role |
|------|------|
| `src/protrider/cooutlier_similarity.py` | Core computation and CSV export |
| `src/protrider/pipeline.py` | `Result.cooutlier_similarity`; compute after `df_Z` |
| `src/protrider/config.py` | Co-outlier config fields and validation |
| `src/protrider/plots.py` | Co-outlier heatmap and embedding plots |
| `src/protrider/cli.py` | Plot subcommands and `plot all` wiring |
| `src/protrider/stability.py` | Disable co-outlier export in stability iterations |
| `src/protrider/__init__.py` | Public API exports |
| `config.yaml` | Documented defaults |
| `README.md` | User-facing overview and examples |
| `tests/test_cooutlier_similarity.py` | Computation and pipeline tests |
| `tests/test_cooutlier_plots.py` | Plot smoke tests |
