from protrider.datasets.covariates import parse_covariates
from protrider.datasets.datasets import ProtriderDataset
from protrider.datasets.protein_intensities import read_protein_intensities
import numpy as np
import pandas as pd


def test_read_protein_intensities(protein_intensities_path, protein_intensities_index_col):
    df = read_protein_intensities(
        protein_intensities_path, protein_intensities_index_col)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'sampleID'
    assert df.shape == (64, 200)


def test_parse_categorical_covariates(categorical_covariates, covariates_path, protein_intensities_path, protein_intensities_index_col):
    """Test basic integration between protein intensities and categorical covariates."""
    protein_intensities = read_protein_intensities(
        protein_intensities_path, protein_intensities_index_col)
    covariates, centered_covariates_noNA = parse_covariates(covariates_path, categorical_covariates)
    assert covariates.shape[0] == protein_intensities.shape[0]
    assert centered_covariates_noNA.shape[0] == protein_intensities.shape[0]


def test_dataset_aligns_covariates_to_intensity_samples(tmp_path):
    """Test that dataset covariates follow intensity sample order, not annotation order."""
    intensities = tmp_path / "intensities.tsv"
    intensities.write_text(
        "sample_1\tsample_2\tprotein_ID\n"
        "1\t2\tprotein_1\n"
        "3\t4\tprotein_2\n"
    )
    annotation = tmp_path / "annotation.tsv"
    annotation.write_text(
        "sample_ID\tAGE\n"
        "sample_2\t20\n"
        "sample_1\t10\n"
    )

    dataset = ProtriderDataset(
        str(intensities),
        "protein_ID",
        sa_file=str(annotation),
        cov_used=["AGE"],
        log_func=None,
        maxNA_filter=1.0,
    )

    np.testing.assert_array_equal(
        dataset.covariates.cpu().numpy()[:, 0],
        np.array([10, 20]),
    )
