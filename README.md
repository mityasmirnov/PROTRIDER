# PROTRIDER

[![Tests](https://github.com/gagneurlab/PROTRIDER/actions/workflows/tests.yml/badge.svg)](https://github.com/gagneurlab/PROTRIDER/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/protrider)](https://pypi.org/project/protrider/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

PROTRIDER is an autoencoder-based method to call protein outliers from mass spectrometry-based proteomics datasets.

Have a look at our [paper](https://doi.org/10.1093/bioinformatics/btaf628) for information about our work.

<details open>
<summary><b>Table of Contents</b></summary>

- [🚀 Quickstart](#-quickstart)
- [⚙️ Installation](#️-installation)
- [📖 Usage](#-usage)
  - [🗂️ Configuration](#️-configuration)
  - [📤 Output](#-output)
  - [🧪 Test latent-space export locally](#-test-latent-space-export-locally)
  - [▶️ Run](#️-run)
- [📄 License](#-license)
- [📚 Citation](#-citation)

</details>

## 🚀 Quickstart

```bash
# 1. Install
pip install protrider

# 2. Run on the included sample data
protrider run --config config.yaml

# 3. Plot results
protrider plot --config config.yaml all
```

Results are written to the directory specified by `out_dir` in `config.yaml` (default: `output/`). The key output file is `protrider_summary.csv`, which contains outlier calls with p-values, z-scores, and fold changes for every sample–protein pair.

## ⚙️ Installation

PROTRIDER was tested using Python 3.14 on Linux. We recommend a dedicated conda environment:

```bash
conda create --name protrider_env python=3.14
conda activate protrider_env
pip install protrider
```

Verify the installation:

```bash
protrider --help
```

More information on conda environments can be found in [Conda's user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/).

To install this fork from GitHub (instead of PyPI), clone [mityasmirnov/PROTRIDER](https://github.com/mityasmirnov/PROTRIDER), then run `uv sync` or `pip install -e .` in the repository root.

## 📖 Usage

### 🗂️ Configuration

All parameters are set in a YAML configuration file. A template is provided as `config.yaml`.

Input files are specified in the config:

- **`input_intensities`**: CSV, TSV, or Parquet file with protein intensities (columns = samples, rows = proteins)
- **`sample_annotation`** (optional): CSV or TSV file with sample covariates (one row per sample)

An example dataset is included under `sample_data/`.

<details open>
<summary><b>Configuration parameters</b></summary>

| Parameter | Description |
|-----------|-------------|
| `out_dir` | Output directory |
| `input_intensities` | Path to protein intensities file |
| `max_allowed_NAs_per_protein` | Maximum percentage of missing values per protein (default: `0.3`) |
| `log_func_name` | Transformation funtion to apply to the data before model fitting: `log` (default), `log10`, `log2`, or `null` (if already log transformed) |
| `sample_annotation` | Path to sample annotations file (optional) |
| `index_col` | Column name containing protein IDs |
| `cov_used` | List of covariate column names from the annotation file (optional) |
| `find_q_method` | Method to determine latent dimension: `OHT` (default), `gs`, `bs` (binary search), or an integer |
| `pval_dist` | Distribution for p-value calculation: `t` (default) or `gaussian` |
| `n_epochs` | Number of training epochs (default: `100`) |
| `checkpoint_path` | Path to save/load model checkpoint (optional) |


</details>

### 📤 Output

The key output file is `protrider_summary.csv`, which contains outlier calls with p-values, z-scores, and fold changes for every sample–protein pair.

<details>
<summary><b>Output files</b></summary>

| File | Description |
|------|-------------|
| `protrider_summary.csv` | Long-format summary with outlier calls for all sample–protein pairs |
| `pvals.csv` | Two-sided p-values (samples × proteins) |
| `pvals_adj.csv` | BH/BY-adjusted p-values |
| `pvals_one_sided.csv` | Left-sided p-values |
| `zscores.csv` | Z-scores |
| `residuals.csv` | Model residuals (observed − predicted) |
| `log2fc.csv` | Log2 fold changes |
| `fc.csv` | Fold changes |
| `output.csv` | Autoencoder reconstructed values |
| `processed_input.csv` | Preprocessed input passed to the autoencoder |
| `additional_info.csv` | Model metadata (latent dimension, learning rate, loss) |
| `train_losses.csv` | Per-epoch training loss |
| `fit_parameters.csv` | Per-protein distribution fit parameters |
| `config.yaml` | Saved configuration for reproducibility |
| `latent_samples.csv` | Sample embeddings in the learned q-dimensional latent space; rows are samples, columns are latent dimensions |
| `latent_protein_loadings_svd.csv` | Protein loadings from SVD/OHT/PCA initialization; rows are proteins, columns are latent dimensions |
| `latent_protein_loadings_decoder.csv` | Learned decoder protein loadings for the linear autoencoder; rows are proteins, columns are latent dimensions |

</details>

<details>
<summary><b>Latent-space outputs</b></summary>

PROTRIDER writes the files below when results are saved in **wide** format (including the default `protrider run --config config.yaml` workflow). They are **not** written for long format (`protrider_summary.csv` only).

| File | Shape | Meaning |
|------|-------|---------|
| `latent_samples.csv` | samples × q | Encoder latent embeddings (use for sample similarity / clustering) |
| `latent_protein_loadings_svd.csv` | proteins × q | Protein loadings from SVD/OHT/PCA init (if `Vt` is available) |
| `latent_protein_loadings_decoder.csv` | proteins × q | Linear decoder loadings (`n_layers: 1` only) |

**Latent samples vs residuals:** `residuals.csv` is observed minus predicted intensities and may remove technical and biological signal the model already explained. `latent_samples.csv` is the encoder representation **before** residualization.

**When decoder loadings are skipped:** multilayer models (`n_layers > 1`) or presence/absence mode — you still get `latent_samples.csv` (and SVD loadings when available).

**OHT with covariates:** the CLI logs a warning that this combination has not been fully evaluated; the run still completes. SVD loadings use the centered protein matrix only (not covariates).

**Downstream example** (sample cosine similarity):

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

Z = pd.read_csv("output/latent_samples.csv", index_col=0)
sim = pd.DataFrame(
    cosine_similarity(Z),
    index=Z.index,
    columns=Z.index,
)
sim.to_csv("output/sample_latent_cosine_similarity.csv")
```

</details>

### 🧪 Test latent-space export locally

Use these steps to verify the feature on your machine **before** merging to `main` or opening a pull request.

#### 1. Get the branch with latent export

```bash
git clone git@github.com:mityasmirnov/PROTRIDER.git
cd PROTRIDER
git checkout feature/latent-space-export
```

If you already have the repo:

```bash
cd PROTRIDER
git fetch origin
git checkout feature/latent-space-export
git pull
```

#### 2. Install dependencies

With [uv](https://github.com/astral-sh/uv) (recommended; uses `uv.lock`):

```bash
uv sync
uv run protrider --help
```

Or with pip in your conda/venv:

```bash
pip install -e .
protrider --help
```

#### 3. Run automated tests

```bash
uv run pytest tests/ -q
```

Optional: only the new latent tests:

```bash
uv run pytest tests/test_latent_space.py -v
```

#### 4. Run the CLI on sample data

From the repository root:

```bash
uv run protrider run --config config.yaml
```

(`config.yaml` points at `sample_data/` and writes to `output/` by default.)

To force a fresh model fit (ignore an old checkpoint):

```bash
rm -f output/model.pt
uv run protrider run --config config.yaml
```

#### 5. Check wide-format outputs

After the run, confirm `output/` contains the usual files **and** the new latent files:

```bash
ls -1 output/latent_*.csv
```

Expected for the default linear config (`n_layers: 1` in `config.yaml`):

- `output/latent_samples.csv`
- `output/latent_protein_loadings_svd.csv`
- `output/latent_protein_loadings_decoder.csv`

Quick shape check (replace `q` with the value in `output/additional_info.csv`):

```bash
python - <<'PY'
import pandas as pd
q = int(pd.read_csv("output/additional_info.csv")["q"].iloc[0])
Z = pd.read_csv("output/latent_samples.csv", index_col=0)
print("latent_samples:", Z.shape, "expect (n_samples, q=", q, ")")
PY
```

#### 6. Optional checks

**Plots still work:**

```bash
uv run protrider plot --config config.yaml all
```

**Python API:**

```python
import protrider

config = protrider.load_config("config.yaml")
result, model_info, fit_params, gs_result = protrider.run(config)
assert result.latent_space is not None
print(result.latent_space.samples.shape)
result.save(config.out_dir, format="wide")
```

**Multilayer** (`n_layers: 2` in config): expect `latent_samples.csv` and SVD loadings, but **no** `latent_protein_loadings_decoder.csv`.

When you are satisfied, merge `feature/latent-space-export` into `main` on your fork (or open a PR to upstream).

### ▶️ Run

Run the pipeline:

```bash
protrider run --config config.yaml
```

Generate plots:

```bash
# All plots
protrider plot --config config.yaml all

# Individual plot types
protrider plot --config config.yaml pvals
protrider plot --config config.yaml aberrant_per_sample
protrider plot --config config.yaml training_loss
protrider plot --config config.yaml encoding_dim

# Expected vs observed for a specific protein
protrider plot --config config.yaml expected_vs_observed --protein_id <protein_id>
```

<details>
<summary><b>Model checkpointing</b></summary>

PROTRIDER automatically saves trained models and reuses them in subsequent runs, skipping retraining if a checkpoint exists. By default the model is saved to `<out_dir>/model.pt`.

To use a custom checkpoint location, set `checkpoint_path` in your config:

```yaml
checkpoint_path: models/my_model.pt
```

To force retraining, delete the checkpoint file or point to a new path.

</details>

<details>
<summary><b>Python API</b></summary>

```python
import protrider

config = protrider.ProtriderConfig(
    out_dir='output/',
    input_intensities='data/protein_intensities.csv',
    sample_annotation='data/sample_annotations.csv',
    index_col='protein_ID',
    cov_used=['AGE', 'SEX'],
    n_epochs=100,
)

# Run
result, model_info, fit_params, gs_result = protrider.run(config)

# Save results
result.save(config.out_dir, format='wide')   # individual CSV files
result.save(config.out_dir, format='long')   # protrider_summary.csv
model_info.save(config.out_dir)
config.save(config.out_dir)

# Generate plots (omit out_dir to get plot objects without saving)
model_info.plot_training_loss(config.out_dir)
result.plot_aberrant_per_sample(config.out_dir)
hist_plot, qq_plot = result.plot_pvals(config.out_dir)
result.plot_expected_vs_observed('protein_123', config.out_dir)

# Access results as DataFrames
result.df_pvals        # p-values
result.df_pvals_adj    # adjusted p-values
result.df_Z            # z-scores
result.df_res          # residuals
result.log2fc          # log2 fold changes
result.fc              # fold changes

# Latent-space objects (also written by result.save(..., format="wide"))
Z_samples = result.latent_space.samples
protein_loadings_svd = result.latent_space.protein_loadings_svd
protein_loadings_decoder = result.latent_space.protein_loadings_decoder  # None for multilayer models
```

</details>

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📚 Citation

If you use PROTRIDER, please cite:

```bibtex
@article{10.1093/bioinformatics/btaf628,
    author = {Klaproth-Andrade, Daniela and Scheller, Ines F and Tsitsiridis, Georgios and Loipfinger, Stefan and Mertes, Christian and Smirnov, Dmitrii and Prokisch, Holger and Yépez, Vicente A and Gagneur, Julien},
    title = {PROTRIDER: Protein abundance outlier detection from mass spectrometry-based proteomics data with a conditional autoencoder},
    journal = {Bioinformatics},
    pages = {btaf628},
    year = {2025},
    month = {11},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf628},
    url = {https://doi.org/10.1093/bioinformatics/btaf628},
}
```
