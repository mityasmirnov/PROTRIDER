# PROTRIDER Nextflow stub

Minimal Nextflow workflow that runs `protrider run` then `protrider plot all` on an intensities TSV, sample annotation, and YAML config. Designed as a local demo using `sample_data/`.

Layout:

- `main.nf` — standalone entry (anonymous workflow)
- `run_plot.nf` — processes + named workflow `PROTRIDER_RUN_PLOT` for include from COWRD umbrella

## Requirements

- Nextflow ≥ 22.10
- `protrider` on `PATH` (e.g. `pip install -e .` or `uv sync` from the repo root)
- Python with PyYAML (`pyaml` / `PyYAML`) for config rewriting

## Quick demo (sample data)

From the repository root:

```bash
# install package into the active environment first
uv sync
# or: pip install -e .

cd nextflow
nextflow run main.nf -profile standard
```

Defaults read:

- `../sample_data/protrider_sample_dataset.tsv`
- `../sample_data/sample_annotations.tsv`
- `../config.yaml`

Outputs land under `../results_nf/`:

- `results/` — PROTRIDER CSV/result artifacts
- `plots/` — plot PNGs from `protrider plot all`

## Custom inputs

```bash
nextflow run main.nf \
  --intensities /path/to/intensities.tsv \
  --sample_annotation /path/to/annotations.tsv \
  --config /path/to/config.yaml \
  --outdir /path/to/outdir \
  --device cpu \
  --n_epochs 20
```

Optional parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--index_col` | `protein_ID` | Protein ID column in the intensities file |
| `--device` | `cpu` | Torch device (`cpu` / `gpu`) |
| `--n_epochs` | `5` | Training epochs (short for the stub) |
| `--find_q_method` | `OHT` | Latent-dimension search method |

Optional `complex_arms` analysis is not enabled by this stub; leave it off unless you wire it separately.

## Notes

- This is a stub/demo, not a production multi-cohort orchestrator.
- Plotting expects a completed `protrider_out/` directory from the run step.
- GPU use requires a CUDA-capable Nextflow executor environment; keep `--device cpu` for the sample demo.
