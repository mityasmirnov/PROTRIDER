#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Named workflow + processes for include from COWRD umbrella (no entry workflow).

process PREPARE_CONFIG {
    tag "config"

    input:
    path base_config
    path intensities
    path sample_annotation

    output:
    path "run_config.yaml"

    script:
    """
    python3 - <<'PY'
import yaml
from pathlib import Path

base = Path("${base_config}")
cfg = yaml.safe_load(base.read_text()) if base.exists() else {}
cfg = cfg or {}
cfg["input_intensities"] = str(Path("${intensities}").resolve())
cfg["sample_annotation"] = str(Path("${sample_annotation}").resolve())
cfg["out_dir"] = "protrider_out"
cfg["index_col"] = "${params.index_col ?: 'protein_ID'}"
cfg["device"] = "${params.device ?: 'cpu'}"
cfg["n_epochs"] = int("${params.n_epochs ?: 5}")
cfg["find_q_method"] = "${params.find_q_method ?: 'OHT'}"
cfg["verbose"] = False
Path("run_config.yaml").write_text(yaml.safe_dump(cfg, default_flow_style=False))
PY
    """
}

process RUN_PROTRIDER {
    tag "protrider-run"
    publishDir "${params.outdir}/protrider_nf/results", mode: "copy"

    input:
    path run_config
    path intensities
    path sample_annotation

    output:
    path "protrider_out/*", emit: results
    path "protrider_out", emit: out_dir

    script:
    """
    protrider run --config ${run_config}
    """
}

process PLOT_PROTRIDER {
    tag "protrider-plot"
    publishDir "${params.outdir}/protrider_nf/plots", mode: "copy"

    input:
    path run_config
    path protrider_out

    output:
    path "protrider_out/plots/*", emit: plots

    script:
    """
    python3 - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${run_config}").read_text())
cfg["out_dir"] = "protrider_out"
Path("plot_config.yaml").write_text(yaml.safe_dump(cfg, default_flow_style=False))
PY
    protrider plot --config plot_config.yaml all
    """
}

workflow PROTRIDER_RUN_PLOT {
    take:
    intensities
    sample_annotation
    base_config

    main:
    prepared = PREPARE_CONFIG(base_config, intensities, sample_annotation)
    run_out = RUN_PROTRIDER(prepared, intensities, sample_annotation)
    plot_out = PLOT_PROTRIDER(prepared, run_out.out_dir)

    emit:
    results = run_out.results
    out_dir = run_out.out_dir
    plots = plot_out.plots
}
