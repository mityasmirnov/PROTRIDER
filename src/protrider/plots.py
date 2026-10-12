import numpy as np
import pandas as pd
import plotnine as pn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from .datasets import covariates


__all__ = [
    "plot_pvals",
    "plot_encoding_dim",
    "plot_aberrant_per_sample",
    "plot_training_loss",
    "plot_expected_vs_observed",
    "plot_correlation_heatmap",
    "plot_patient_similarity",
    "plot_cooutlier_patient_similarity",
    "plot_cooutlier_patient_pca",
    "plot_cooutlier_patient_umap",
    "plot_cooutlier_patient_tsne",
    "plot_patient_latent_pca",
    "plot_patient_latent_umap",
    "plot_patient_latent_tsne",
    "plot_volcano",
    "plot_intensity_rank",
    "plot_gene_sample_heatmap",
    "plot_cv_before_after",
]

logger = logging.getLogger(__name__)


def plot_pvals(output_dir=None, distribution="t", plot_title="", fontsize=10, pvals_one_sided=None):
    """
    Plot p-value distributions.
    
    Args:
        output_dir: Optional directory to save plots. If None, plots are not saved.
        distribution: Distribution type ("t" or "normal")
        plot_title: Title for the plots
        fontsize: Font size for plot text
        pvals_one_sided: Optional DataFrame with p-values. If None, reads from output_dir/pvals_one_sided.csv
        
    Returns:
        tuple: (histogram_plot, qq_plot) - plotnine plot objects
    """
    if pvals_one_sided is None:
        if output_dir is None:
            raise ValueError("Either output_dir or pvals_one_sided must be provided")
        dt_pvals = pd.read_csv(os.path.join(output_dir, "pvals_one_sided.csv")).set_index('proteinID')
    else:
        dt_pvals = pvals_one_sided.copy()
    
    dt_pvals = dt_pvals.melt()
    dt_pvals = dt_pvals.dropna(subset=['value'])
    
    if distribution == "t":
        dist = "Student's t-distribution"
        p_color = "#0047ab"
    else:
        dist = "Normal distribution"
        p_color = "#3cb371"

    dt_pvals['type'] = dist

    g_pvals = (
        pn.ggplot(dt_pvals, pn.aes(x='value')) +
        pn.geom_histogram(bins=100, boundary=0, fill=p_color) +
        pn.theme_bw(base_size=fontsize) +
        pn.facet_wrap('~type', ncol=1) +
        pn.labs(
            x='One-sided P-value',
            y='Count',
            title=plot_title
        ) 
    )
    
    if output_dir is not None:
        os.makedirs(f"{output_dir}/plots/", exist_ok=True)
        g_pvals.save(f"{output_dir}/plots/pvalues_dist.png", width=4, height=4, dpi=300)

    pv_t = dt_pvals['value'].dropna().values
    theoretical = -np.log10((np.arange(1, len(pv_t) + 1)) / (len(pv_t) + 1))
    sample = -np.log10(np.sort(pv_t))
    qq_data = pd.DataFrame({
        'Theoretical': theoretical,
        'Sample': sample,
        'type': dist
    })

    g_qq = (
        pn.ggplot(qq_data, pn.aes(x='Theoretical', y='Sample')) +
        pn.geom_point(color=p_color) +
        pn.geom_abline(intercept=0, slope=1, color='lightgrey', linetype='dashed') +
        pn.labs(
            title=plot_title,
            x='Theoretical -log10(P-value)',
            y='Observed -log10(P-value)'
        ) +
        pn.theme_bw(base_size=fontsize)
    )

    if output_dir is not None:
        g_qq.save(f"{output_dir}/plots/qqplots.png", width=4, height=4, dpi=300)
    
    return g_pvals, g_qq


def plot_encoding_dim(output_dir, find_q_method, plot_title="", oht_q=None, fontsize=10):
    os.makedirs(f"{output_dir}/plots/", exist_ok=True)
 
    if find_q_method != "gs":
        print("plot_encoding_dim is not implemented for OHT yet.")
        return
    
    encoding_dims = pd.read_csv(output_dir + "/grid_search.csv")
    additional_info = pd.read_csv(output_dir + "/additional_info.csv")
    bestQ = additional_info["q"].iloc[0]
    best_row = encoding_dims[encoding_dims["encod_dim"] == bestQ]

    p_out = (
        pn.ggplot(encoding_dims, pn.aes("encod_dim", "aucpr")) +
        pn.geom_point() +
        pn.scale_x_log10() +
        #pn.geom_smooth(method='loess') +
        # Optimum line & label
        pn.geom_vline(pn.aes(xintercept="encod_dim", color="'Grid search'"), 
                   data=best_row, show_legend=True) +
        pn.geom_text(pn.aes(x=bestQ - 0.5, y=0.0, label=str(bestQ)), data=best_row) +
        pn.labs(x="Encoding dimensions", 
                y="Evaluation loss (AUPRC)", 
                color="Optimal Q", 
                title="Search for best encoding dimension " + plot_title)
    )

    # Add OHT line/label if present
    if oht_q is not None:
        p_out += (
            pn.geom_vline(pn.aes(xintercept=oht_q, color="'OHT'"), show_legend=True) +
            pn.geom_text(pn.aes(x=oht_q - 0.5, y=0.0, label=str(oht_q)),
                      data=pd.DataFrame({"x": [oht_q], "y": [0.0]})) +
            pn.scale_color_manual(values={"Grid search": "red", "OHT": "lightblue"})
        )
    else:
        p_out += pn.scale_color_manual(values={"Grid search": "red"})
    
    p_out += pn.theme_bw(base_size=fontsize)

    p_out.save(f"{output_dir}/plots/encoding_dim_search.png",
               width=6, height=4, units='in', dpi=300)


def plot_aberrant_per_sample(output_dir=None, plot_title="", fontsize=10, protrider_summary=None):
    """
    Plot aberrant protein counts per sample.
    
    Args:
        output_dir: Optional directory to save plot. If None, plot is not saved.
        plot_title: Title for the plot
        fontsize: Font size for plot text
        protrider_summary: Optional DataFrame with summary. If None, reads from output_dir/protrider_summary.csv
        
    Returns:
        plotnine plot object
    """
    if protrider_summary is None:
        if output_dir is None:
            raise ValueError("Either output_dir or protrider_summary must be provided")
        res = pd.read_csv(f"{output_dir}/protrider_summary.csv")
    else:
        res = protrider_summary.copy()

    aberrants = res[res["PROTEIN_outlier"]]
    non_aberrant_ids = set(res["sampleID"].unique()) - set(aberrants["sampleID"])
    non_aberrant = pd.DataFrame({"sampleID": list(non_aberrant_ids), "outlier_count": 0})

    # Count aberrant outliers per sample
    aberrant_counts = aberrants["sampleID"].value_counts().reset_index()
    aberrant_counts.columns = ["sampleID", "outlier_count"]

    # Combine and sort
    aberrant_numbers = pd.concat([aberrant_counts, non_aberrant], ignore_index=True)
    aberrant_numbers = aberrant_numbers.sort_values("outlier_count").reset_index(drop=True)
    aberrant_numbers["Rank"] = aberrant_numbers.index + 1
    
    median_val = aberrant_numbers["outlier_count"].median()
    percentile_95 = aberrant_numbers["outlier_count"].quantile(0.95)

    yadjust = 1.2
    
    p_out = (
        pn.ggplot(aberrant_numbers, pn.aes(x="Rank", y="outlier_count + 1")) +
        pn.geom_line(color="blue") +
        pn.labs(x="Sample rank", y="Outliers per sample + 1", title=plot_title) +
        pn.scale_y_log10() +
        pn.geom_hline(yintercept=median_val, color="black") +
        pn.geom_hline(yintercept=percentile_95, color="black") +
        pn.geom_text(
            pn.aes(x=5, y=median_val*yadjust, label="'Median'"),
            size=fontsize*0.7,
            ha='left'
        ) +
        pn.geom_text(
            pn.aes(x=5, y=percentile_95*yadjust, label="'95th percentile'"),
            size=fontsize*0.7,
            ha='left'
        ) +
        pn.theme_bw(base_size=fontsize)
    )
    
    if output_dir is not None:
        os.makedirs(f"{output_dir}/plots/", exist_ok=True)
        p_out.save(f"{output_dir}/plots/aberrant_per_sample.png",
                   width=6, height=4, units='in', dpi=300)
    
    return p_out


def plot_training_loss(output_dir=None, plot_title="", fontsize=10, train_losses=None):
    """
    Plot training loss over epochs.
    
    Args:
        output_dir: Optional directory to save plot. If None, plot is not saved.
        plot_title: Title for the plot
        fontsize: Font size for plot text
        train_losses: Optional DataFrame with training losses. If None, reads from output_dir/train_losses.csv
        
    Returns:
        plotnine plot object
    """
    if train_losses is None:
        if output_dir is None:
            raise ValueError("Either output_dir or train_losses must be provided")
        loss_histoty = pd.read_csv(f"{output_dir}/train_losses.csv")
    else:
        loss_histoty = train_losses.copy()
    
    p_out = (
        pn.ggplot(loss_histoty, pn.aes(x="epoch", y="train_loss")) +
        pn.geom_line(color="blue") +
        pn.labs(x="Epoch", y="Loss", title=plot_title) +
        pn.theme_bw(base_size=fontsize) +
        pn.scale_y_log10()
    )

    if output_dir is not None:
        os.makedirs(f"{output_dir}/plots/", exist_ok=True)
        p_out.save(f"{output_dir}/plots/training_loss.png",
                   width=6, height=4, units='in', dpi=300)
    
    return p_out


def plot_correlation_heatmap(output_dir, sample_annotation_path: str, plot_title="", covariate_name=None):
    """
    Create a correlation heatmap plot for protein data colored by covariate values.
    
    Args:
        
    """
    output_dir = Path(output_dir)
    zscore_data = pd.read_csv(output_dir / 'zscores.csv').set_index('proteinID')

    row_colors = None
    if sample_annotation_path is not None:
        sample_annotation = covariates.read_annotation_file(sample_annotation_path)
        # Get covariate values for coloring
        if covariate_name is not None:
            if covariate_name not in sample_annotation.columns:
                raise ValueError(f"Covariate '{covariate_name}' not found in sample annotation.")
            covariate_values = sample_annotation[covariate_name]
            
            # Create a lookup table for coloring rows
            # Use a larger color palette to support more unique covariate values
            unique_vals = covariate_values.unique()
            if len(unique_vals) > 20:
                logger.warning(f"Covariate '{covariate_name}' has more than 20 unique values. Skipping color annotation.")
            else:
                palette = sns.color_palette("tab20", len(unique_vals))
                lut = dict(zip(unique_vals, palette))
                row_colors = [lut[label] for label in covariate_values]
        
    # Calculate correlation matrix
    corr_matrix = zscore_data.corr()
    
    # Create clustermap
    clustermap = sns.clustermap(
        corr_matrix,
        cmap='mako',
        row_colors=None if not row_colors else row_colors,
    )
    
    # Add legend for row colors if they exist
    if row_colors is not None and covariate_name is not None:
        # Create legend patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=str(val)) 
                          for val, color in lut.items()]
        clustermap.ax_col_dendrogram.legend(handles=legend_elements, 
                                           title=covariate_name,
                                           bbox_to_anchor=(1.15, 1), 
                                           loc='upper left',
                                           frameon=True)
    
    # Adjust layout
    plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)
    plt.title(plot_title)
    plt.savefig(output_dir / 'plots' / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correlation heatmap to {output_dir / 'plots' / 'correlation_heatmap.png'}")

def plot_expected_vs_observed(protein_id, output_dir=None, plot_title="", fontsize=10, 
                             processed_input=None, output_data=None, protrider_summary=None):
    """
    Plot expected vs observed intensities for a specific protein.
    
    Args:
        protein_id: ID of the protein to plot
        output_dir: Optional directory to save plot. If None, plot is not saved.
        plot_title: Title for the plot
        fontsize: Font size for plot text
        processed_input: Optional DataFrame with processed input data. If None, reads from output_dir/processed_input.csv
        output_data: Optional DataFrame with output data. If None, reads from output_dir/output.csv
        protrider_summary: Optional DataFrame with summary. If None, reads from output_dir/protrider_summary.csv
        
    Returns:
        plotnine plot object
    """
    if processed_input is None or output_data is None or protrider_summary is None:
        if output_dir is None:
            raise ValueError("Either output_dir or all data DataFrames must be provided")
        data_in = pd.read_csv(f"{output_dir}/processed_input.csv")
        data_out = pd.read_csv(f"{output_dir}/output.csv")
        summary = pd.read_csv(f"{output_dir}/protrider_summary.csv")
    else:
        data_in = processed_input.copy()
        data_out = output_data.copy()
        summary = protrider_summary.copy()

    data_in_melted = data_in[data_in["proteinID"] == protein_id].melt(id_vars="proteinID", var_name="SAMPLE_ID", value_name="in_int")
    data_out_melted = data_out[data_out["proteinID"] == protein_id].melt(id_vars="proteinID", var_name="SAMPLE_ID", value_name="out_int")

    # Merge inputs and outputs
    df_plot = pd.merge(data_in_melted, data_out_melted, on="SAMPLE_ID")
    summary_filtered = summary[(summary["proteinID"] == protein_id) & summary["PROTEIN_outlier"]]
    is_outlier = df_plot["SAMPLE_ID"].isin(summary_filtered["sampleID"])
    df_plot["outlier"] = is_outlier.replace({True: "Outlier", False: None})
    df_plot["label"] = df_plot["SAMPLE_ID"].where(is_outlier, None)
    df_plot["outlier"] = pd.Categorical(df_plot["outlier"], categories=["Outlier"])

    p_out = (
        pn.ggplot(df_plot, pn.aes(x="out_int", y="in_int")) +
        pn.geom_point(pn.aes(color="outlier"), show_legend=False) +
        pn.geom_text(
            pn.aes(label="label"),
            size=4,
            va="bottom", ha="left",
            nudge_y=0.001  # slight vertical shift to avoid overlap
        ) +
        pn.scale_x_log10() +
        pn.scale_y_log10() +
        pn.labs(
            x="Expected log10-intensity",
            y="Observed log10-intensity",
            title=f"Protein: {protein_id}"
        ) +
        pn.theme_bw(base_size=fontsize)
    )

    if output_dir is not None:
        os.makedirs(f"{output_dir}/plots/", exist_ok=True)
        p_out.save(f"{output_dir}/plots/expected_vs_observed.png",
                   width=4, height=4, units='in', dpi=300)
    
    return p_out


def plot_patient_similarity(
    output_dir=None,
    plot_title="",
    patient_similarity=None,
    patient_subpopulations=None,
):
    """
    Plot patient/sample similarity heatmap from latent-derived RBF similarities.

    Reads patient_similarity.csv (and optional patient_subpopulations.csv) when
    DataFrames are not passed. Returns None with a warning if inputs are missing.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)

    if patient_similarity is None:
        if output_dir is None:
            raise ValueError("Either output_dir or patient_similarity must be provided")
        similarity_path = output_dir / "patient_similarity.csv"
        if not similarity_path.exists():
            logger.warning(
                "Skipping patient similarity plot: %s not found", similarity_path
            )
            return None
        patient_similarity = pd.read_csv(similarity_path, index_col=0)

    if patient_subpopulations is None and output_dir is not None:
        subpop_path = output_dir / "patient_subpopulations.csv"
        if subpop_path.exists():
            patient_subpopulations = pd.read_csv(subpop_path)

    if output_dir is None:
        raise ValueError("output_dir is required to save the patient similarity plot")

    os.makedirs(output_dir / "plots", exist_ok=True)
    similarity = patient_similarity.astype(float)

    row_colors = None
    lut = None
    if patient_subpopulations is not None and "subpopulation" in patient_subpopulations.columns:
        subpop_map = patient_subpopulations.set_index("sampleID")["subpopulation"]
        unique_vals = subpop_map.unique()
        if len(unique_vals) <= 20:
            palette = sns.color_palette("tab20", len(unique_vals))
            lut = dict(zip(unique_vals, palette))
            row_colors = [lut[subpop_map.get(idx, unique_vals[0])] for idx in similarity.index]
        else:
            logger.warning(
                "More than 20 subpopulations; skipping row color annotation on heatmap"
            )

    plot_obj = None
    try:
        if similarity.shape[0] >= 3:
            plot_obj = sns.clustermap(
                similarity,
                cmap="mako",
                vmin=0,
                vmax=1,
                row_colors=row_colors,
                col_colors=row_colors,
            )
            if row_colors is not None and lut is not None:
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor=color, label=str(val)) for val, color in lut.items()
                ]
                plot_obj.ax_col_dendrogram.legend(
                    handles=legend_elements,
                    title="subpopulation",
                    bbox_to_anchor=(1.15, 1),
                    loc="upper left",
                    frameon=True,
                )
            if plot_title:
                plot_obj.fig.suptitle(plot_title)
            plot_obj.savefig(
                output_dir / "plots" / "patient_similarity_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            raise ValueError("Too few samples for clustermap")
    except Exception as exc:
        logger.warning("Clustermap failed (%s); falling back to heatmap", exc)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(similarity, cmap="mako", vmin=0, vmax=1, ax=ax)
        if plot_title:
            ax.set_title(plot_title)
        fig.savefig(
            output_dir / "plots" / "patient_similarity_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plot_obj = fig

    plt.close("all")
    logger.info(
        "Saved patient similarity heatmap to %s",
        output_dir / "plots" / "patient_similarity_heatmap.png",
    )
    return plot_obj


def _plot_patient_embedding(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: str,
    fontsize: int,
) -> object:
    """Shared plotnine scatter for patient latent 2D embeddings."""
    plot_df = df.copy()
    plot_df["subpopulation"] = plot_df["subpopulation"].astype(str)
    p_out = (
        pn.ggplot(plot_df, pn.aes(x=x, y=y, color="subpopulation"))
        + pn.geom_point(size=3)
        + pn.theme_bw(base_size=fontsize)
        + pn.labs(x=x, y=y, color="Subpopulation", title=title)
    )
    p_out.save(output_path, width=6, height=4, units="in", dpi=300)
    return p_out


def _plot_patient_latent_embedding_from_csv(
    output_dir,
    plot_title: str,
    fontsize: int,
    csv_name: str,
    x_col: str,
    y_col: str,
    png_name: str,
    default_title: str,
    embedding_df=None,
):
    if embedding_df is None:
        if output_dir is None:
            raise ValueError(f"Either output_dir or embedding data must be provided for {csv_name}")
        output_dir = Path(output_dir)
        csv_path = output_dir / csv_name
        if not csv_path.exists():
            logger.warning("Skipping %s plot: %s not found", default_title, csv_path)
            return None
        embedding_df = pd.read_csv(csv_path)
    elif output_dir is not None:
        output_dir = Path(output_dir)

    if output_dir is None:
        raise ValueError(f"output_dir is required to save the {default_title} plot")

    os.makedirs(output_dir / "plots", exist_ok=True)
    output_path = output_dir / "plots" / png_name
    p_out = _plot_patient_embedding(
        embedding_df,
        x_col,
        y_col,
        output_path,
        plot_title or default_title,
        fontsize,
    )
    logger.info("Saved %s plot to %s", default_title, output_path)
    return p_out


def plot_patient_latent_pca(
    output_dir=None,
    plot_title="",
    fontsize=10,
    patient_latent_pca=None,
):
    """
    Scatter plot of PCA coordinates colored by inferred subpopulation.

    Reads patient_latent_pca.csv when a DataFrame is not passed. Returns None
    with a warning if the file is missing.
    """
    return _plot_patient_latent_embedding_from_csv(
        output_dir,
        plot_title,
        fontsize,
        csv_name="patient_latent_pca.csv",
        x_col="PC1",
        y_col="PC2",
        png_name="patient_latent_pca.png",
        default_title="Patient latent PCA",
        embedding_df=patient_latent_pca,
    )


def plot_patient_latent_umap(
    output_dir=None,
    plot_title="",
    fontsize=10,
    patient_latent_umap=None,
):
    """
    Scatter plot of UMAP coordinates colored by inferred subpopulation.

    Reads patient_latent_umap.csv when a DataFrame is not passed. Returns None
    with a warning if the file is missing.
    """
    return _plot_patient_latent_embedding_from_csv(
        output_dir,
        plot_title,
        fontsize,
        csv_name="patient_latent_umap.csv",
        x_col="UMAP1",
        y_col="UMAP2",
        png_name="patient_latent_umap.png",
        default_title="Patient latent UMAP",
        embedding_df=patient_latent_umap,
    )


def plot_patient_latent_tsne(
    output_dir=None,
    plot_title="",
    fontsize=10,
    patient_latent_tsne=None,
):
    """
    Scatter plot of t-SNE coordinates colored by inferred subpopulation.

    Reads patient_latent_tsne.csv when a DataFrame is not passed. Returns None
    with a warning if the file is missing.
    """
    return _plot_patient_latent_embedding_from_csv(
        output_dir,
        plot_title,
        fontsize,
        csv_name="patient_latent_tsne.csv",
        x_col="TSNE1",
        y_col="TSNE2",
        png_name="patient_latent_tsne.png",
        default_title="Patient latent t-SNE",
        embedding_df=patient_latent_tsne,
    )


def plot_cooutlier_patient_similarity(
    output_dir=None,
    plot_title="",
    patient_similarity=None,
    patient_subpopulations=None,
):
    """
    Plot co-outlier directional Jaccard similarity heatmap.

    Reads cooutlier_patient_similarity.csv (and optional subpopulations) when
    DataFrames are not passed. Returns None with a warning if inputs are missing.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)

    if patient_similarity is None:
        if output_dir is None:
            raise ValueError("Either output_dir or patient_similarity must be provided")
        similarity_path = output_dir / "cooutlier_patient_similarity.csv"
        if not similarity_path.exists():
            logger.warning(
                "Skipping co-outlier patient similarity plot: %s not found",
                similarity_path,
            )
            return None
        patient_similarity = pd.read_csv(similarity_path, index_col=0)

    if patient_subpopulations is None and output_dir is not None:
        subpop_path = output_dir / "cooutlier_patient_subpopulations.csv"
        if subpop_path.exists():
            patient_subpopulations = pd.read_csv(subpop_path)

    if output_dir is None:
        raise ValueError(
            "output_dir is required to save the co-outlier patient similarity plot"
        )

    os.makedirs(output_dir / "plots", exist_ok=True)
    similarity = patient_similarity.astype(float)

    row_colors = None
    lut = None
    if (
        patient_subpopulations is not None
        and "subpopulation" in patient_subpopulations.columns
    ):
        subpop_map = patient_subpopulations.set_index("sampleID")["subpopulation"]
        unique_vals = subpop_map.unique()
        if len(unique_vals) <= 20:
            palette = sns.color_palette("tab20", len(unique_vals))
            lut = dict(zip(unique_vals, palette))
            row_colors = [
                lut[subpop_map.get(idx, unique_vals[0])] for idx in similarity.index
            ]
        else:
            logger.warning(
                "More than 20 subpopulations; skipping row color annotation on heatmap"
            )

    plot_obj = None
    try:
        if similarity.shape[0] >= 3:
            plot_obj = sns.clustermap(
                similarity,
                cmap="mako",
                vmin=0,
                vmax=1,
                row_colors=row_colors,
                col_colors=row_colors,
            )
            if row_colors is not None and lut is not None:
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor=color, label=str(val))
                    for val, color in lut.items()
                ]
                plot_obj.ax_col_dendrogram.legend(
                    handles=legend_elements,
                    title="subpopulation",
                    bbox_to_anchor=(1.15, 1),
                    loc="upper left",
                    frameon=True,
                )
            if plot_title:
                plot_obj.fig.suptitle(plot_title)
            plot_obj.savefig(
                output_dir / "plots" / "cooutlier_patient_similarity_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            raise ValueError("Too few samples for clustermap")
    except Exception as exc:
        logger.warning("Co-outlier clustermap failed (%s); falling back to heatmap", exc)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(similarity, cmap="mako", vmin=0, vmax=1, ax=ax)
        if plot_title:
            ax.set_title(plot_title)
        fig.savefig(
            output_dir / "plots" / "cooutlier_patient_similarity_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plot_obj = fig

    plt.close("all")
    logger.info(
        "Saved co-outlier patient similarity heatmap to %s",
        output_dir / "plots" / "cooutlier_patient_similarity_heatmap.png",
    )
    return plot_obj


def _plot_cooutlier_embedding_from_csv(
    output_dir,
    plot_title: str,
    fontsize: int,
    csv_name: str,
    x_col: str,
    y_col: str,
    png_name: str,
    default_title: str,
    embedding_df=None,
):
    if embedding_df is None:
        if output_dir is None:
            raise ValueError(
                f"Either output_dir or embedding data must be provided for {csv_name}"
            )
        output_dir = Path(output_dir)
        csv_path = output_dir / csv_name
        if not csv_path.exists():
            logger.warning("Skipping %s plot: %s not found", default_title, csv_path)
            return None
        embedding_df = pd.read_csv(csv_path)
    elif output_dir is not None:
        output_dir = Path(output_dir)

    if output_dir is None:
        raise ValueError(f"output_dir is required to save the {default_title} plot")

    plot_df = embedding_df.dropna(subset=[x_col, y_col])
    if len(plot_df) < 2:
        logger.warning(
            "Skipping %s plot: fewer than two samples with finite coordinates",
            default_title,
        )
        return None

    os.makedirs(output_dir / "plots", exist_ok=True)
    output_path = output_dir / "plots" / png_name
    p_out = _plot_patient_embedding(
        plot_df,
        x_col,
        y_col,
        output_path,
        plot_title or default_title,
        fontsize,
    )
    logger.info("Saved %s plot to %s", default_title, output_path)
    return p_out


def plot_cooutlier_patient_pca(
    output_dir=None,
    plot_title="",
    fontsize=10,
    cooutlier_patient_pca=None,
):
    """Scatter plot of co-outlier TruncatedSVD coordinates colored by subpopulation."""
    return _plot_cooutlier_embedding_from_csv(
        output_dir,
        plot_title,
        fontsize,
        csv_name="cooutlier_patient_pca.csv",
        x_col="PC1",
        y_col="PC2",
        png_name="cooutlier_patient_pca.png",
        default_title="Co-outlier patient PCA",
        embedding_df=cooutlier_patient_pca,
    )


def plot_cooutlier_patient_umap(
    output_dir=None,
    plot_title="",
    fontsize=10,
    cooutlier_patient_umap=None,
):
    """Scatter plot of co-outlier UMAP coordinates colored by subpopulation."""
    return _plot_cooutlier_embedding_from_csv(
        output_dir,
        plot_title,
        fontsize,
        csv_name="cooutlier_patient_umap.csv",
        x_col="UMAP1",
        y_col="UMAP2",
        png_name="cooutlier_patient_umap.png",
        default_title="Co-outlier patient UMAP",
        embedding_df=cooutlier_patient_umap,
    )


def plot_cooutlier_patient_tsne(
    output_dir=None,
    plot_title="",
    fontsize=10,
    cooutlier_patient_tsne=None,
):
    """Scatter plot of co-outlier t-SNE coordinates colored by subpopulation."""
    return _plot_cooutlier_embedding_from_csv(
        output_dir,
        plot_title,
        fontsize,
        csv_name="cooutlier_patient_tsne.csv",
        x_col="TSNE1",
        y_col="TSNE2",
        png_name="cooutlier_patient_tsne.png",
        default_title="Co-outlier patient t-SNE",
        embedding_df=cooutlier_patient_tsne,
    )


def _read_protein_by_sample_matrix(path: Path) -> pd.DataFrame:
    """Load a wide protein×sample CSV (proteinID index, sample columns)."""
    df = pd.read_csv(path)
    if "proteinID" in df.columns:
        df = df.set_index("proteinID")
    else:
        df = df.set_index(df.columns[0])
    return df


def _load_protrider_summary(output_dir=None, protrider_summary=None) -> pd.DataFrame:
    if protrider_summary is not None:
        return protrider_summary.copy()
    if output_dir is None:
        raise ValueError("Either output_dir or protrider_summary must be provided")
    summary_path = Path(output_dir) / "protrider_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"protrider_summary.csv not found at {summary_path}")
    return pd.read_csv(summary_path)


def _default_sample_id(summary: pd.DataFrame) -> str:
    if "PROTEIN_outlier" in summary.columns:
        outliers = summary[summary["PROTEIN_outlier"].astype(bool)]
        if len(outliers) > 0:
            return str(outliers["sampleID"].iloc[0])
    return str(summary["sampleID"].iloc[0])


def _default_protein_id(summary: pd.DataFrame) -> str:
    if "PROTEIN_outlier" in summary.columns:
        outliers = summary[summary["PROTEIN_outlier"].astype(bool)]
        if len(outliers) > 0:
            return str(outliers["proteinID"].iloc[0])
    return str(summary["proteinID"].iloc[0])


def _protein_cv(matrix: pd.DataFrame) -> pd.Series:
    """Coefficient of variation per protein (columns = samples)."""
    values = matrix.astype(float)
    means = values.mean(axis=1, skipna=True).abs()
    stds = values.std(axis=1, skipna=True, ddof=0)
    cv = stds / means.replace(0, np.nan)
    return cv.replace([np.inf, -np.inf], np.nan).dropna()


def plot_volcano(
    output_dir=None,
    sample_id=None,
    plot_title="",
    fontsize=10,
    y_metric="pvalue",
    protrider_summary=None,
):
    """
    Volcano plot for one sample (OUTRIDER plotVolcano analogue).

    X is effect size (PROTEIN_LOG2FC). Y is either -log10(p-value) or |z-score|.

    Args:
        output_dir: Directory to save plots under plots/. Optional if DataFrame given.
        sample_id: Sample to plot. Defaults to first outlier sample, else first sample.
        plot_title: Optional title suffix.
        fontsize: Base font size.
        y_metric: "pvalue" (default) or "zscore".
        protrider_summary: Optional long-format summary DataFrame.

    Returns:
        plotnine plot object, or None if inputs are missing.
    """
    try:
        summary = _load_protrider_summary(output_dir, protrider_summary)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("Skipping volcano plot: %s", exc)
        return None

    if sample_id is None:
        sample_id = _default_sample_id(summary)

    df = summary[summary["sampleID"].astype(str) == str(sample_id)].copy()
    if df.empty:
        logger.warning("Skipping volcano plot: sample_id %s not found", sample_id)
        return None

    required = {"PROTEIN_LOG2FC", "PROTEIN_PVALUE", "PROTEIN_ZSCORE"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Skipping volcano plot: missing columns %s", sorted(missing))
        return None

    df = df.dropna(subset=["PROTEIN_LOG2FC", "PROTEIN_PVALUE", "PROTEIN_ZSCORE"])
    if df.empty:
        logger.warning("Skipping volcano plot: no finite values for sample %s", sample_id)
        return None

    if y_metric == "zscore":
        df["y_value"] = df["PROTEIN_ZSCORE"].abs()
        y_label = "|Z-score|"
    elif y_metric == "pvalue":
        p = df["PROTEIN_PVALUE"].clip(lower=np.finfo(float).tiny)
        df["y_value"] = -np.log10(p)
        y_label = "-log10(P-value)"
    else:
        raise ValueError("y_metric must be 'pvalue' or 'zscore'")

    if "PROTEIN_outlier" in df.columns:
        df["status"] = np.where(
            df["PROTEIN_outlier"].astype(bool), "Outlier", "Non-outlier"
        )
    else:
        df["status"] = "Non-outlier"

    title = plot_title or f"Volcano: {sample_id}"
    p_out = (
        pn.ggplot(df, pn.aes(x="PROTEIN_LOG2FC", y="y_value", color="status"))
        + pn.geom_point(alpha=0.7, size=1.5)
        + pn.scale_color_manual(values={"Outlier": "#d62728", "Non-outlier": "#1f77b4"})
        + pn.labs(
            x="log2 fold change",
            y=y_label,
            color="",
            title=title,
        )
        + pn.theme_bw(base_size=fontsize)
    )

    if output_dir is not None:
        out = Path(output_dir)
        os.makedirs(out / "plots", exist_ok=True)
        p_out.save(
            out / "plots" / "volcano.png",
            width=6,
            height=4,
            units="in",
            dpi=300,
        )
        logger.info("Saved volcano plot to %s", out / "plots" / "volcano.png")

    return p_out


def plot_intensity_rank(
    output_dir=None,
    protein_id=None,
    plot_title="",
    fontsize=10,
    protrider_summary=None,
):
    """
    Intensity rank plot for one protein (OUTRIDER plotExpressionRank analogue).

    Samples are ranked by observed log-intensity; outliers are highlighted.

    Returns:
        plotnine plot object, or None if inputs are missing.
    """
    try:
        summary = _load_protrider_summary(output_dir, protrider_summary)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("Skipping intensity rank plot: %s", exc)
        return None

    if protein_id is None:
        protein_id = _default_protein_id(summary)

    df = summary[summary["proteinID"].astype(str) == str(protein_id)].copy()
    if df.empty:
        logger.warning(
            "Skipping intensity rank plot: protein_id %s not found", protein_id
        )
        return None

    intensity_col = None
    for candidate in ("PROTEIN_LOG2INT", "PROTEIN_INT", "PROTEIN_EXPECTED_LOG2INT"):
        if candidate in df.columns:
            intensity_col = candidate
            break
    if intensity_col is None:
        logger.warning(
            "Skipping intensity rank plot: no intensity column in protrider_summary"
        )
        return None

    df = df.dropna(subset=[intensity_col]).sort_values(intensity_col)
    if df.empty:
        logger.warning(
            "Skipping intensity rank plot: no finite intensities for %s", protein_id
        )
        return None

    df = df.reset_index(drop=True)
    df["Rank"] = df.index + 1
    if "PROTEIN_outlier" in df.columns:
        df["status"] = np.where(
            df["PROTEIN_outlier"].astype(bool), "Outlier", "Non-outlier"
        )
    else:
        df["status"] = "Non-outlier"

    title = plot_title or f"Intensity rank: {protein_id}"
    p_out = (
        pn.ggplot(df, pn.aes(x="Rank", y=intensity_col, color="status"))
        + pn.geom_point(size=2)
        + pn.scale_color_manual(values={"Outlier": "#d62728", "Non-outlier": "#1f77b4"})
        + pn.labs(
            x="Sample rank",
            y=intensity_col.replace("_", " "),
            color="",
            title=title,
        )
        + pn.theme_bw(base_size=fontsize)
    )

    if output_dir is not None:
        out = Path(output_dir)
        os.makedirs(out / "plots", exist_ok=True)
        p_out.save(
            out / "plots" / "intensity_rank.png",
            width=6,
            height=4,
            units="in",
            dpi=300,
        )
        logger.info(
            "Saved intensity rank plot to %s", out / "plots" / "intensity_rank.png"
        )

    return p_out


def plot_gene_sample_heatmap(
    output_dir=None,
    plot_title="",
    n_proteins=50,
    processed_input=None,
    output_data=None,
):
    """
    Protein×sample heatmaps for raw (processed input) and AE-normalized output.

    Analogue of OUTRIDER plotCountGeneSampleHeatmap. Selects the top ``n_proteins``
    by coefficient of variation on the processed input when more proteins are present.

    Returns:
        tuple of (raw_plot, normalized_plot), either may be None.
    """
    out = Path(output_dir) if output_dir is not None else None

    if processed_input is None:
        if out is None:
            raise ValueError("Either output_dir or processed_input must be provided")
        raw_path = out / "processed_input.csv"
        if not raw_path.exists():
            logger.warning(
                "Skipping gene-sample heatmap: %s not found", raw_path
            )
            return None, None
        processed_input = _read_protein_by_sample_matrix(raw_path)
    else:
        processed_input = processed_input.copy()
        if "proteinID" in getattr(processed_input, "columns", []):
            processed_input = processed_input.set_index("proteinID")

    if output_data is None and out is not None:
        norm_path = out / "output.csv"
        if norm_path.exists():
            output_data = _read_protein_by_sample_matrix(norm_path)
        else:
            logger.warning(
                "Normalized output.csv not found; plotting raw heatmap only"
            )
    elif output_data is not None:
        output_data = output_data.copy()
        if "proteinID" in getattr(output_data, "columns", []):
            output_data = output_data.set_index("proteinID")

    if out is None:
        raise ValueError("output_dir is required to save gene-sample heatmaps")

    os.makedirs(out / "plots", exist_ok=True)

    cv = _protein_cv(processed_input)
    if len(cv) == 0:
        logger.warning("Skipping gene-sample heatmap: no finite CV values")
        return None, None

    top_ids = cv.sort_values(ascending=False).head(n_proteins).index
    raw_sub = processed_input.loc[processed_input.index.isin(top_ids)].astype(float)

    raw_plot = None
    norm_plot = None

    try:
        g = sns.clustermap(
            raw_sub.fillna(raw_sub.median(axis=0)),
            cmap="mako",
            figsize=(8, max(4, len(raw_sub) * 0.12 + 2)),
            col_cluster=True,
            row_cluster=True,
        )
        title = plot_title or "Protein × sample (processed input)"
        g.fig.suptitle(title)
        g.savefig(out / "plots" / "gene_sample_heatmap_raw.png", dpi=300, bbox_inches="tight")
        raw_plot = g
        plt.close("all")
        logger.info(
            "Saved raw gene-sample heatmap to %s",
            out / "plots" / "gene_sample_heatmap_raw.png",
        )
    except Exception as exc:
        logger.warning("Raw clustermap failed (%s); falling back to heatmap", exc)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(raw_sub, cmap="mako", ax=ax)
        ax.set_title(plot_title or "Protein × sample (processed input)")
        fig.savefig(
            out / "plots" / "gene_sample_heatmap_raw.png",
            dpi=300,
            bbox_inches="tight",
        )
        raw_plot = fig
        plt.close("all")

    if output_data is not None:
        shared = raw_sub.index.intersection(output_data.index)
        if len(shared) > 0:
            norm_sub = output_data.loc[shared].astype(float)
            try:
                g = sns.clustermap(
                    norm_sub.fillna(norm_sub.median(axis=0)),
                    cmap="mako",
                    figsize=(8, max(4, len(norm_sub) * 0.12 + 2)),
                    col_cluster=True,
                    row_cluster=True,
                )
                title = plot_title or "Protein × sample (AE output)"
                g.fig.suptitle(title)
                g.savefig(
                    out / "plots" / "gene_sample_heatmap_normalized.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                norm_plot = g
                plt.close("all")
                logger.info(
                    "Saved normalized gene-sample heatmap to %s",
                    out / "plots" / "gene_sample_heatmap_normalized.png",
                )
            except Exception as exc:
                logger.warning(
                    "Normalized clustermap failed (%s); falling back to heatmap", exc
                )
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(norm_sub, cmap="mako", ax=ax)
                ax.set_title(plot_title or "Protein × sample (AE output)")
                fig.savefig(
                    out / "plots" / "gene_sample_heatmap_normalized.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                norm_plot = fig
                plt.close("all")

    return raw_plot, norm_plot


def plot_cv_before_after(
    output_dir=None,
    plot_title="",
    fontsize=10,
    processed_input=None,
    residuals=None,
):
    """
    Coefficient of variation before vs after AE (OUTRIDER BCV analogue).

    Before: CV of processed input intensities per protein.
    After: CV of AE residuals per protein (residual std / |mean input|).

    Returns:
        plotnine plot object, or None if inputs are missing.
    """
    out = Path(output_dir) if output_dir is not None else None

    if processed_input is None:
        if out is None:
            raise ValueError("Either output_dir or processed_input must be provided")
        raw_path = out / "processed_input.csv"
        if not raw_path.exists():
            logger.warning("Skipping CV before/after plot: %s not found", raw_path)
            return None
        processed_input = _read_protein_by_sample_matrix(raw_path)
    else:
        processed_input = processed_input.copy()
        if "proteinID" in getattr(processed_input, "columns", []):
            processed_input = processed_input.set_index("proteinID")

    if residuals is None:
        if out is None:
            raise ValueError("Either output_dir or residuals must be provided")
        res_path = out / "residuals.csv"
        if not res_path.exists():
            logger.warning("Skipping CV before/after plot: %s not found", res_path)
            return None
        residuals = _read_protein_by_sample_matrix(res_path)
    else:
        residuals = residuals.copy()
        if "proteinID" in getattr(residuals, "columns", []):
            residuals = residuals.set_index("proteinID")

    before_cv = _protein_cv(processed_input)
    means = processed_input.astype(float).mean(axis=1, skipna=True).abs().replace(0, np.nan)
    shared = before_cv.index.intersection(residuals.index).intersection(means.index)
    if len(shared) == 0:
        logger.warning("Skipping CV before/after plot: no shared proteins")
        return None

    after_std = residuals.loc[shared].astype(float).std(axis=1, skipna=True, ddof=0)
    after_cv = (after_std / means.loc[shared]).replace([np.inf, -np.inf], np.nan).dropna()
    before_cv = before_cv.loc[after_cv.index]

    cv_df = pd.concat(
        [
            pd.DataFrame({"when": "Before", "CV": before_cv.values}),
            pd.DataFrame({"when": "After", "CV": after_cv.values}),
        ],
        ignore_index=True,
    )
    cv_df["when"] = pd.Categorical(cv_df["when"], categories=["Before", "After"])

    title = plot_title or "Coefficient of variation before/after AE"
    p_out = (
        pn.ggplot(cv_df, pn.aes(x="when", y="CV"))
        + pn.geom_boxplot(fill="#aec7e8")
        + pn.labs(
            x="Autoencoder correction",
            y="Coefficient of variation",
            title=title,
        )
        + pn.theme_bw(base_size=fontsize)
    )

    if out is not None:
        os.makedirs(out / "plots", exist_ok=True)
        p_out.save(
            out / "plots" / "cv_before_after.png",
            width=5,
            height=4,
            units="in",
            dpi=300,
        )
        logger.info(
            "Saved CV before/after plot to %s", out / "plots" / "cv_before_after.png"
        )

    return p_out

