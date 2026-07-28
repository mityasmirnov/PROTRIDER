from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
import logging


logger = logging.getLogger(__name__)

__all__ = ['NoValidCovariatesError', 'parse_covariates']

_SAMPLE_ID_COLUMN_CANDIDATES = ("sample_ID", "sampleID", "sample_id")


class NoValidCovariatesError(ValueError):
    """Raised when all requested covariates are unusable after parsing."""


def parse_covariates(
    sa_file: Optional[str],
    cov_used: Optional[list],
    sample_ids: Optional[Iterable[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse covariates from sample annotation file.
    
    Args:
        sa_file: Path to sample annotation file (CSV/TSV)
        cov_used: List of covariate column names to use
        sample_ids: Optional sample IDs from the intensity matrix. When
            provided, annotation rows are aligned to this order by sample ID.
        
    Returns:
        tuple: (covariates, centered_covariates_noNA)
            - covariates: Raw covariates with NAs replaced by 0
            - centered_covariates_noNA: Centered numerical covariates with NAs replaced by 0
    """
    if sa_file is None:
        raise ValueError("Sample annotation file is required.")
    if cov_used is None:
        raise ValueError("Covariates to use must be specified.")

    # Read sample annotation file
    sample_anno = read_annotation_file(sa_file)
    logger.info(f'Finished reading sample annotation with shape: {sample_anno.shape}')
    if sample_ids is not None:
        sample_anno = _align_annotation_to_samples(sample_anno, sample_ids)
    
    # Process covariates
    processed_covariates = _process_covariates(sample_anno[cov_used])
    
    # Combine all covariate types
    covariates, centered_covariates = _combine_covariates(processed_covariates)
    
    # Validate output
    assert np.isnan(covariates).sum() == 0, "Covariates contain NaN values"
    assert np.isnan(centered_covariates).sum() == 0, "Centered covariates contain NaN values"
    
    return covariates, centered_covariates


def read_annotation_file(sa_file):
    """Read sample annotation file based on file extension."""
    file_extension = Path(sa_file).suffix
    if file_extension == '.csv':
        return pd.read_csv(sa_file)
    elif file_extension == '.tsv':
        return pd.read_csv(sa_file, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def _align_annotation_to_samples(sample_anno, sample_ids):
    """Return annotation rows in the same order as the intensity sample IDs."""
    id_col = _detect_sample_id_column(sample_anno)
    if id_col is None:
        expected_columns = ", ".join(_SAMPLE_ID_COLUMN_CANDIDATES)
        raise ValueError(
            "Sample annotation must include a sample ID column "
            f"({expected_columns}) when covariates are used."
        )

    expected_ids = [str(sample_id) for sample_id in sample_ids]
    if pd.Index(expected_ids).duplicated().any():
        duplicated = pd.Index(expected_ids)[pd.Index(expected_ids).duplicated()].unique()
        raise ValueError(f"Duplicate sample IDs in intensity data: {duplicated.tolist()[:5]}...")

    indexed = sample_anno.copy()
    indexed[id_col] = indexed[id_col].astype(str)
    duplicated = indexed.loc[indexed[id_col].duplicated(), id_col].unique()
    if len(duplicated) > 0:
        raise ValueError(f"Duplicate sample IDs in annotation '{id_col}': {duplicated.tolist()[:5]}...")

    indexed = indexed.set_index(id_col, drop=False)
    missing = [sample_id for sample_id in expected_ids if sample_id not in indexed.index]
    if missing:
        raise ValueError(
            f"Samples missing from annotation '{id_col}': {missing[:5]}..."
        )

    extra_count = len(set(indexed.index) - set(expected_ids))
    if extra_count:
        logger.warning(
            "Ignoring %d annotation rows not present in intensity data.",
            extra_count,
        )

    return indexed.loc[expected_ids].reset_index(drop=True)


def _detect_sample_id_column(sample_anno):
    for candidate in _SAMPLE_ID_COLUMN_CANDIDATES:
        if candidate in sample_anno.columns:
            return candidate
    return None


def _is_numeric_dtype(dtype):
    """Check if pandas dtype is numeric."""
    return pd.api.types.is_numeric_dtype(dtype)


def _process_covariates(cov_data):
    """Process covariates into numerical, categorical, and NA indicator variables."""
    numerical_covs = []
    categorical_covs = []
    na_indicator_covs = []
    
    for col_name, series in cov_data.items():
        # Skip covariates with insufficient variation
        if series.nunique() < 2:
            logger.warning(f"Covariate '{col_name}' has less than 2 unique values. Skipping...")
            continue
        
        # Create NA indicator if needed
        if series.isna().any():
            na_indicator = series.isna().astype(int).to_frame(name=f'{col_name}_NA')
            na_indicator_covs.append(na_indicator)
        
        # Process based on data type
        if _is_numeric_dtype(series.dtype) and series.dtype != bool:
            numerical_covs.append(series.to_frame())
        else:
            # One-hot encode categorical variables
            categorical_encoded = pd.get_dummies(series, drop_first=True, dummy_na=False, prefix=col_name)
            categorical_covs.append(categorical_encoded)
    
    # Log covariate counts
    logger.info(f'No. numerical covariates used: {len(numerical_covs)}')
    logger.info(f'No. categorical covariates used: {len(categorical_covs)}')
    logger.info(f'No. NA indicator covariates used: {len(na_indicator_covs)}')
    
    return {
        'numerical': numerical_covs,
        'categorical': categorical_covs,
        'na_indicators': na_indicator_covs
    }


def _combine_covariates(processed_covariates):
    """Combine processed covariates into final arrays."""
    covariate_arrays = []
    centered_covariate_arrays = []
    
    # Process numerical covariates
    if processed_covariates['numerical']:
        numerical_data = pd.concat(processed_covariates['numerical'], axis=1).values
        # Center numerical data and handle NAs
        means = np.nanmean(numerical_data, axis=0, keepdims=True)
        centered_numerical = numerical_data - means
        
        # Replace NAs with 0
        numerical_no_na = np.where(np.isnan(numerical_data), 0, numerical_data)
        centered_numerical_no_na = np.where(np.isnan(centered_numerical), 0, centered_numerical)
        
        covariate_arrays.append(numerical_no_na)
        centered_covariate_arrays.append(centered_numerical_no_na)
    
    # Process categorical covariates (no centering needed)
    if processed_covariates['categorical']:
        categorical_data = pd.concat(processed_covariates['categorical'], axis=1).values
        covariate_arrays.append(categorical_data)
        centered_covariate_arrays.append(categorical_data)
    
    # Process NA indicators (no centering needed)
    if processed_covariates['na_indicators']:
        na_data = pd.concat(processed_covariates['na_indicators'], axis=1).values
        covariate_arrays.append(na_data)
        centered_covariate_arrays.append(na_data)
    
    # Concatenate all covariate types
    if covariate_arrays:
        covariates = np.concatenate(covariate_arrays, axis=1)
        centered_covariates = np.concatenate(centered_covariate_arrays, axis=1)
    else:
        raise NoValidCovariatesError("No valid covariates found.")
    
    return covariates, centered_covariates