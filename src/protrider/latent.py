from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch

from .datasets import ProtriderDataset, ProtriderSubset
from .model import ProtriderAutoencoder

logger = logging.getLogger(__name__)

__all__ = ["LatentSpace", "extract_latent_space"]


def _latent_column_names(q: int) -> list[str]:
    return [f"latent_{i + 1}" for i in range(q)]


@dataclass
class LatentSpace:
    """Sample latent embeddings and optional protein loading matrices."""

    samples: pd.DataFrame
    protein_loadings_svd: Optional[pd.DataFrame] = None
    protein_loadings_decoder: Optional[pd.DataFrame] = None

    def save(self, out_dir: str) -> dict[str, Path]:
        """Write latent-space CSV files to out_dir and return paths written."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        samples_path = out_path / "latent_samples.csv"
        self.samples.to_csv(samples_path, header=True, index=True)
        written["latent_samples"] = samples_path
        logger.info("Saved latent sample embeddings to %s", samples_path)

        if self.protein_loadings_svd is not None:
            svd_path = out_path / "latent_protein_loadings_svd.csv"
            self.protein_loadings_svd.to_csv(svd_path, header=True, index=True)
            written["latent_protein_loadings_svd"] = svd_path
            logger.info("Saved SVD protein loadings to %s", svd_path)

        if self.protein_loadings_decoder is not None:
            dec_path = out_path / "latent_protein_loadings_decoder.csv"
            self.protein_loadings_decoder.to_csv(dec_path, header=True, index=True)
            written["latent_protein_loadings_decoder"] = dec_path
            logger.info("Saved decoder protein loadings to %s", dec_path)

        return written


def extract_latent_space(
    dataset: Union[ProtriderDataset, ProtriderSubset],
    model: ProtriderAutoencoder,
    q: int,
) -> LatentSpace:
    """Extract sample latent embeddings and optional protein loadings from a fitted model."""
    q = int(q)
    latent_cols = _latent_column_names(q)

    with torch.no_grad():
        z_sample = (
            model.encoder(dataset.X, cond=dataset.covariates).detach().cpu().numpy()
        )

    df_samples = pd.DataFrame(
        z_sample,
        index=dataset.data.index,
        columns=latent_cols,
    )

    df_protein_loadings_svd = None
    if getattr(dataset, "Vt", None) is not None:
        n_proteins = dataset.data.shape[1]
        vt_q = dataset.Vt[:q, :n_proteins]
        df_protein_loadings_svd = pd.DataFrame(
            vt_q.T,
            index=dataset.data.columns,
            columns=latent_cols,
        )
    else:
        logger.warning(
            "dataset.Vt is not available; skipping latent_protein_loadings_svd.csv"
        )

    df_protein_loadings_decoder = None
    if model.n_layers == 1 and not model.presence_absence:
        w_decoder = model.decoder.model.weight.detach().cpu().numpy()
        w_decoder_latent = w_decoder[:, :q]
        df_protein_loadings_decoder = pd.DataFrame(
            w_decoder_latent,
            index=dataset.data.columns,
            columns=latent_cols,
        )
    elif model.n_layers > 1:
        logger.warning(
            "Skipping latent_protein_loadings_decoder.csv: multilayer model "
            "(n_layers=%s) has no single linear protein x latent loading matrix",
            model.n_layers,
        )
    elif model.presence_absence:
        logger.warning(
            "Skipping latent_protein_loadings_decoder.csv: presence/absence mode "
            "does not define a standard linear decoder loading matrix"
        )

    return LatentSpace(
        samples=df_samples,
        protein_loadings_svd=df_protein_loadings_svd,
        protein_loadings_decoder=df_protein_loadings_decoder,
    )
