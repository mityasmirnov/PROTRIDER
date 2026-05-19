#'---
#' title: Protrider Genecode OHT Normal distribution new protrider
#' author: Dmitrii Smirnov
#'---

suppressPackageStartupMessages({
  library(data.table)
  library(yaml)
})

# ============================================================
# Absolute paths — edit these
# ============================================================

conda_env <- "omicsDiagnosticsDev"

repo_root <- "/Users/Mitya/Desktop/working/PROTRIDER"
base_config_path <- file.path(repo_root, "rwd", "config_protrider_dev.yml")

input_intensities <- "/Users/Mitya/Desktop/working/omicsDagnostics_data/processed_data/protrider/protrider_data.tsv"
sample_annotation <- "/Users/Mitya/Desktop/working/omicsDagnostics_data/processed_data/protrider/protrider_annotation.tsv"

out_dir <- "/Volumes/Transcend/prot/protrider_stability_dev"
 
run_config_path <- file.path(out_dir, "config_protrider_stability_run.yaml")

standard_rds_path <- file.path(out_dir, "PROTRIDER_results_stability.rds")
stability_rds_path <- file.path(out_dir, "PROTRIDER_results_bs.rds")

# If TRUE, pauses after each preview plot (interactive R only).
pause_between_plots <- FALSE

# If TRUE, run a fast smoke test (5 epochs, 2 stability runs).
smoke_test <- FALSE

# If TRUE, skip protrider CLI and only load outputs + make stability plots.
# Override: PROTRIDER_SKIP_RUN=1 Rscript tests/PROTRIDER_OHT_N_Pro.R
skip_protrider_run <- identical(Sys.getenv("PROTRIDER_SKIP_RUN", "0"), "1")

# ============================================================
# Stability settings
# ============================================================

cohort_stability_n_runs <- 10      # start small for testing; later use 100
cohort_stability_min_runs <- 10    # later use 30
cohort_stability_drop_fraction <- 0.1
cohort_stability_min_samples <- 30
cohort_stability_seed <- 42

# ============================================================
# Helpers
# ============================================================

stop_if_missing <- function(path, label) {
  if (!file.exists(path)) {
    stop(label, " does not exist: ", path)
  }
}



preview_pause <- function() {
  if (isTRUE(pause_between_plots) && interactive()) {
    readline(prompt = "Press [Enter] for next plot...")
  }
}

# ============================================================
# Checks
# ============================================================

stop_if_missing(base_config_path, "Base PROTRIDER dev config")
stop_if_missing(input_intensities, "Input intensities")
stop_if_missing(sample_annotation, "Sample annotation")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Build run-specific PROTRIDER config
# ============================================================

config_list <- yaml::read_yaml(base_config_path)

config_list$out_dir <- out_dir
config_list$input_intensities <- input_intensities
config_list$sample_annotation <- sample_annotation

# omicsDiagnostics input format:
# rows = proteins/genes, columns = samples, first column = geneID
config_list$index_col <- "geneID"
config_list$input_format <- "proteins_as_rows"

# Stability mode should use OHT.
config_list$find_q_method <- "OHT"

# Keep all sample-protein pairs in protrider_summary.csv.
config_list$report_all <- TRUE

# Stability analysis (use integer scalars so YAML is not written as 30.0)
config_list$cohort_stability <- TRUE
config_list$cohort_stability_n_runs <- as.integer(cohort_stability_n_runs)
config_list$cohort_stability_min_runs <- as.integer(cohort_stability_min_runs)
config_list$cohort_stability_max_runtime_min <- NULL
config_list$cohort_stability_drop_fraction <- cohort_stability_drop_fraction
config_list$cohort_stability_min_samples <- as.integer(cohort_stability_min_samples)
config_list$cohort_stability_seed <- as.integer(cohort_stability_seed)
config_list$cohort_stability_require_oht <- TRUE
config_list$cohort_stability_save_iteration_files <- FALSE

# Apple Silicon / M1 Mac: use CPU (no CUDA). See rwd/config_protrider_dev.yml.
config_list$device <- "cpu"
config_list$n_jobs <- -1

if (isTRUE(smoke_test)) {
  config_list$n_epochs <- 5L
  config_list$patience <- 3L
  config_list$cohort_stability_n_runs <- 2L
  config_list$cohort_stability_min_runs <- 2L
}

# Optional outputs used for downstream tables.
# These CSVs are still produced by PROTRIDER, but this script will not save plots.
config_list$export_latent_space <- TRUE
config_list$export_patient_similarity <- TRUE
config_list$export_cooutlier_patient_similarity <- TRUE

# Co-outlier defaults
if (is.null(config_list$z_threshold)) {
  config_list$z_threshold <- 3.0
}
if (is.null(config_list$cooutlier_min_anomalies)) {
  config_list$cooutlier_min_anomalies <- 1
}
if (is.null(config_list$cooutlier_max_clusters)) {
  config_list$cooutlier_max_clusters <- 10
}
if (is.null(config_list$cooutlier_min_samples_for_clustering)) {
  config_list$cooutlier_min_samples_for_clustering <- 4
}

# Avoid accidental checkpoint reuse from old runs.
config_list$checkpoint_path <- NULL

yaml::write_yaml(config_list, run_config_path)

message("Wrote run-specific stability config to: ", run_config_path)
message("Output directory: ", out_dir)

# ============================================================
# Run PROTRIDER
# ============================================================

run_checked <- function(cmd, args) {
  message("\nRunning command:\n", paste(c(cmd, args), collapse = " "))
  
  status <- system2(cmd, args = args)
  
  if (!identical(status, 0L)) {
    stop(
      "Command failed with exit status ", status, ":\n",
      paste(c(cmd, args), collapse = " ")
    )
  }
  
  invisible(status)
}


if (!isTRUE(skip_protrider_run)) {
  run_checked(
    "conda",
    c(
      "run", "-n", conda_env,
      "protrider", "run", "--config", run_config_path
    )
  )

  run_checked(
    "conda",
    c(
      "run", "-n", conda_env,
      "protrider", "plot", "--config", run_config_path, "all"
    )
  )

  message("\nPROTRIDER stability run finished.")
  message("Plots saved under: ", file.path(out_dir, "plots"))
} else {
  message("Skipping protrider run (skip_protrider_run=TRUE).")
}

# ============================================================
# Load standard output
# ============================================================

summary_path <- file.path(out_dir, "protrider_summary.csv")
stop_if_missing(summary_path, "protrider_summary.csv")

res <- fread(summary_path)

if (all(c("sampleID", "proteinID") %in% names(res))) {
  setnames(res, c("sampleID", "proteinID"), c("SAMPLE_ID", "geneID"))
}

saveRDS(res, standard_rds_path)
message("Saved standard PROTRIDER result RDS: ", standard_rds_path)

# ============================================================
# Load stability output
# ============================================================

bs_path <- file.path(out_dir, "protrider_summary_bs.csv")

if (!file.exists(bs_path)) {
  warning(
    "protrider_summary_bs.csv was not produced. ",
    "This usually means cohort size <= 30 or stability mode was skipped."
  )
  bs <- NULL
} else {
  bs <- fread(bs_path)
  
  if (all(c("sampleID", "proteinID") %in% names(bs))) {
    setnames(bs, c("sampleID", "proteinID"), c("SAMPLE_ID", "geneID"))
  }
  
  saveRDS(bs, stability_rds_path)
  message("Saved stability result RDS: ", stability_rds_path)
}

# ============================================================
# Stability QC plots (saved to plots/ when possible)
# ============================================================

plots_dir <- file.path(out_dir, "plots")
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

save_stability_plot <- function(expr, filename, width = 8, height = 6) {
  path <- file.path(plots_dir, filename)
  png(path, width = width, height = height, units = "in", res = 150)
  on.exit(dev.off(), add = TRUE)
  eval(expr)
  message("Saved stability plot: ", path)
}

if (!is.null(bs)) {
  
  # ------------------------------------------------------------
  # 1. Histogram: outlier call rate
  # ------------------------------------------------------------
  
  if ("PROTEIN_outlier_call_rate" %in% names(bs)) {
    save_stability_plot(
      quote({
        hist(
          bs$PROTEIN_outlier_call_rate,
          breaks = 50,
          main = "Outlier call stability",
          xlab = "Outlier call rate across observed stability runs",
          col = "grey80",
          border = "grey30"
        )
        abline(v = c(0.5, 0.8), lty = 2, lwd = 2)
      }),
      "stability_outlier_call_rate.png"
    )
    preview_pause()
  }
  
  # ------------------------------------------------------------
  # 2. Histogram: observed fraction
  # ------------------------------------------------------------
  
  if ("BS_OBSERVED_FRACTION" %in% names(bs)) {
    save_stability_plot(
      quote({
        hist(
          bs$BS_OBSERVED_FRACTION,
          breaks = 50,
          main = "Sample-protein pair availability",
          xlab = "Fraction of completed stability runs where pair was observed",
          col = "grey80",
          border = "grey30"
        )
      }),
      "stability_observed_fraction.png"
    )
    preview_pause()
  }
  
  # ------------------------------------------------------------
  # 3. Histogram: FC interval width
  # ------------------------------------------------------------
  
  if (all(c("PROTEIN_FC_q025", "PROTEIN_FC_q975") %in% names(bs))) {
    bs[, PROTEIN_FC_interval_width := PROTEIN_FC_q975 - PROTEIN_FC_q025]
    
    save_stability_plot(
      quote({
        hist(
          bs$PROTEIN_FC_interval_width,
          breaks = 50,
          main = "Fold-change stability interval width",
          xlab = "PROTEIN_FC_q975 - PROTEIN_FC_q025",
          col = "grey80",
          border = "grey30"
        )
      }),
      "stability_fc_interval_width.png"
    )
    preview_pause()
  }
  
  # ------------------------------------------------------------
  # 4. Scatter: full FC vs call rate
  # ------------------------------------------------------------
  
  if (all(c("PROTEIN_FC_full", "PROTEIN_outlier_call_rate") %in% names(bs))) {
    save_stability_plot(
      quote({
        plot(
          x = bs$PROTEIN_FC_full,
          y = bs$PROTEIN_outlier_call_rate,
          pch = 16,
          cex = 0.45,
          xlab = "Full-cohort PROTEIN_FC",
          ylab = "Outlier call rate",
          main = "Full-run fold change vs stability call rate"
        )
        abline(h = c(0.5, 0.8), lty = 2, lwd = 2)
        abline(v = 1, lty = 3, lwd = 2)
      }),
      "stability_fc_vs_call_rate.png",
      width = 8,
      height = 7
    )
    preview_pause()
  }
  
  # ------------------------------------------------------------
  # 5. Scatter: full log2FC vs call rate
  # ------------------------------------------------------------
  
  if (all(c("PROTEIN_LOG2FC_full", "PROTEIN_outlier_call_rate") %in% names(bs))) {
    save_stability_plot(
      quote({
        plot(
          x = bs$PROTEIN_LOG2FC_full,
          y = bs$PROTEIN_outlier_call_rate,
          pch = 16,
          cex = 0.45,
          xlab = "Full-cohort PROTEIN_LOG2FC",
          ylab = "Outlier call rate",
          main = "Full-run log2FC vs stability call rate"
        )
        abline(h = c(0.5, 0.8), lty = 2, lwd = 2)
        abline(v = 0, lty = 3, lwd = 2)
      }),
      "stability_log2fc_vs_call_rate.png",
      width = 8,
      height = 7
    )
    preview_pause()
  }
  
  # ------------------------------------------------------------
  # 6. Preview unstable full-run outliers as a table
  # ------------------------------------------------------------
  
  needed_cols <- c(
    "PROTEIN_outlier_full",
    "PROTEIN_outlier_call_rate",
    "PROTEIN_FC_full",
    "PROTEIN_FC_q025",
    "PROTEIN_FC_q975"
  )
  
  if (all(needed_cols %in% names(bs))) {
    unstable <- bs[
      PROTEIN_outlier_full == TRUE &
        !is.na(PROTEIN_outlier_call_rate) &
        PROTEIN_outlier_call_rate < 0.5
    ]
    
    message("\nUnstable full-run outliers with call_rate < 0.5: ", nrow(unstable))
    
    if (nrow(unstable) > 0) {
      print(
        unstable[
          order(PROTEIN_outlier_call_rate)
        ][
          1:min(.N, 20),
          .(
            SAMPLE_ID,
            geneID,
            PROTEIN_FC_full,
            PROTEIN_FC_q025,
            PROTEIN_FC_q975,
            PROTEIN_outlier_call_rate,
            BS_N_OBSERVED,
            BS_OBSERVED_FRACTION
          )
        ]
      )
    }
  }
}

# ============================================================
# Optional preview of latent / patient outputs
# ============================================================

latent_path <- file.path(out_dir, "latent_samples.csv")
patient_subpop_path <- file.path(out_dir, "patient_subpopulations.csv")
cooutlier_subpop_path <- file.path(out_dir, "cooutlier_patient_subpopulations.csv")

if (file.exists(latent_path)) {
  latent <- fread(latent_path)
  
  message("\nlatent_samples.csv preview:")
  print(head(latent))
}

if (file.exists(patient_subpop_path)) {
  patient_subpop <- fread(patient_subpop_path)
  
  message("\npatient_subpopulations.csv preview:")
  print(head(patient_subpop))
}

if (file.exists(cooutlier_subpop_path)) {
  cooutlier_subpop <- fread(cooutlier_subpop_path)
  
  message("\ncooutlier_patient_subpopulations.csv preview:")
  print(head(cooutlier_subpop))
}

message("\nDone.")