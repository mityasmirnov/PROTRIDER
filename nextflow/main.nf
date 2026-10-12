#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.intensities = "${projectDir}/../sample_data/protrider_sample_dataset.tsv"
params.sample_annotation = "${projectDir}/../sample_data/sample_annotations.tsv"
params.config = "${projectDir}/../config.yaml"
params.outdir = "${projectDir}/../results_nf"
params.index_col = "protein_ID"
params.device = "cpu"
params.n_epochs = 5
params.find_q_method = "OHT"

include { PROTRIDER_RUN_PLOT } from './run_plot.nf'

workflow {
    intensities_ch = Channel.fromPath(params.intensities, checkIfExists: true)
    annotation_ch = Channel.fromPath(params.sample_annotation, checkIfExists: true)
    config_ch = Channel.fromPath(params.config, checkIfExists: true)

    PROTRIDER_RUN_PLOT(intensities_ch, annotation_ch, config_ch)
}
