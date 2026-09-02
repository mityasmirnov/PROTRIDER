"""Configuration loading utilities."""
import yaml
import torch
from pathlib import Path
from typing import Union, Callable
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import numpy as np

@dataclass
class ProtriderConfig:
    """Configuration for PROTRIDER pipeline.
    
    Matrix Format Requirements:
    ---------------------------
    input_intensities: File path (str)
        - File format: columns = samples, rows = proteins
    
    sample_annotation: File path (str) or None
        - Format: rows = samples
    """
    
    # I/O paths
    input_intensities: str  # File path only
    input_format: Literal["proteins_as_rows", "proteins_as_columns"] = "proteins_as_rows"
    index_col: str = "protein_ID"
    out_dir: Optional[str] = None  # File path or None
    sample_annotation: Optional[str] = None  # File path or None
    
    # Preprocessing params
    max_allowed_NAs_per_protein: float = 0.3
    log_func_name: Optional[Literal["log", "log2", "log10"]] = "log"
    
    # Computed fields (set in __post_init__)
    log_func: Optional[Callable] = field(init=False, repr=False, default=None)
    base_fn: Callable = field(init=False, repr=False, default=None)
    device_torch: torch.device = field(init=False, repr=False, default=None)
    
    # Covariates
    cov_used: Optional[List[str]] = None
    
    # Reproducibility
    seed: Optional[int] = 42
    
    # Grid search outlier injection params
    inj_freq: float = 1e-3
    inj_mean: float = 3
    inj_sd: float = 1.6
    gs_epochs: int = None
    
    # Model params
    autoencoder_training: bool = True
    n_layers: int = 1
    n_epochs: int = 100
    lr: float = 1e-4
    batch_size: Optional[int] = None
    find_q_method: Union[str, int] = "OHT"  # "OHT", "gs", or an integer
    init_pca: bool = True
    h_dim: Optional[int] = None
    patience: int = 50
    min_delta: float = 1e-4
    
    # Presence absence modelling
    presence_absence: bool = False
    lambda_presence_absence: float = 0.5

    # Wandb logging
    use_wandb: bool = False
    wandb_project: Optional[str] = "protrider"
    wandb_name: Optional[str] = None
    
    # Statistical params
    pval_dist: Literal["gaussian", "t"] = "t"
    pval_adj: Literal["by", "bh"] = "by"
    pval_sided: Literal["two-sided", "left", "right"] = "two-sided"
    pseudocount: float = 0.01
    common_degrees_freedom: bool = True  # whether to use common degrees of freedom; more stable for small datasets
    
    # Reporting params
    outlier_threshold: float = 0.1
    report_all: bool = True
    
    # Runtime params
    verbose: bool = False
    device: Literal["gpu", "cpu"] = "gpu"
    n_jobs: int = -1  # Number of parallel jobs, -1 means using all processors
    
    # Model checkpoint path for saving/loading trained models
    # If None: train from scratch and save to out_dir/model.pt
    # If path exists: load model from this path and skip training
    # If path doesn't exist: train and save to this path
    checkpoint_path: Optional[str] = None

    # Cohort stability analysis (subsampling; not classical bootstrap)
    cohort_stability: bool = False
    cohort_stability_n_runs: int = 100
    cohort_stability_min_runs: int = 30
    cohort_stability_max_runtime_min: Optional[float] = None
    cohort_stability_drop_fraction: float = 0.1
    cohort_stability_min_samples: int = 30
    cohort_stability_seed: Optional[int] = None
    cohort_stability_require_oht: bool = True
    cohort_stability_save_iteration_files: bool = False

    # Optional exports (disabled in cohort stability iterations for speed)
    export_latent_space: bool = True
    export_patient_similarity: bool = True

    # Co-outlier patient stratification
    export_cooutlier_patient_similarity: bool = True
    z_threshold: float = 3.0
    cooutlier_min_anomalies: int = 1
    cooutlier_max_clusters: int = 10
    cooutlier_min_samples_for_clustering: int = 4
    
    def __post_init__(self):
        """Validate configuration after initialization and set computed fields."""
        self.find_q_method = _coerce_find_q_method(self.find_q_method)

        # Validation
        if self.max_allowed_NAs_per_protein < 0 or self.max_allowed_NAs_per_protein > 1:
            raise ValueError("max_allowed_NAs_per_protein must be between 0 and 1")
        
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        
        if self.outlier_threshold < 0 or self.outlier_threshold > 1:
            raise ValueError("outlier_threshold must be between 0 and 1")
        
        if self.find_q_method not in ["OHT", "gs", "bs"] and not self.find_q_method.isdigit():
            raise ValueError("find_q_method must be 'OHT', 'gs', 'bs' or an integer string")
        
        if self.presence_absence and self.n_layers != 1:
            import warnings
            warnings.warn("Presence absence modeling is only validated with n_layers=1")

        if self.cohort_stability_n_runs < 1:
            raise ValueError("cohort_stability_n_runs must be at least 1")
        if self.cohort_stability_min_runs < 1:
            raise ValueError("cohort_stability_min_runs must be at least 1")
        if self.cohort_stability_min_runs > self.cohort_stability_n_runs:
            raise ValueError(
                "cohort_stability_min_runs must be <= cohort_stability_n_runs"
            )
        if not (0 < self.cohort_stability_drop_fraction < 1):
            raise ValueError(
                "cohort_stability_drop_fraction must be between 0 and 1 (exclusive)"
            )
        if self.cohort_stability_min_samples < 2:
            raise ValueError("cohort_stability_min_samples must be at least 2")
        if (
            self.cohort_stability_max_runtime_min is not None
            and self.cohort_stability_max_runtime_min <= 0
        ):
            raise ValueError(
                "cohort_stability_max_runtime_min must be None or positive"
            )
        if (
            self.cohort_stability
            and self.cohort_stability_require_oht
            and self.find_q_method != "OHT"
        ):
            raise ValueError(
                "cohort_stability requires find_q_method='OHT' when "
                "cohort_stability_require_oht is True"
            )

        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if self.cooutlier_min_anomalies < 0:
            raise ValueError("cooutlier_min_anomalies must be >= 0")
        if self.cooutlier_max_clusters < 2:
            raise ValueError("cooutlier_max_clusters must be >= 2")
        if self.cooutlier_min_samples_for_clustering < 2:
            raise ValueError("cooutlier_min_samples_for_clustering must be >= 2")
        
        # Set log_func and base_fn based on log_func_name
        if self.log_func_name == "log2":
            self.log_func = np.log2
            self.base_fn = lambda x: 2 ** x
        elif self.log_func_name == "log10":
            self.log_func = np.log10
            self.base_fn = lambda x: 10 ** x
        elif self.log_func_name == "log":
            self.log_func = np.log
            self.base_fn = np.exp
        elif self.log_func_name is None:
            self.log_func = None
            self.base_fn = np.exp
        else:
            raise ValueError(f"Log func {self.log_func_name} not supported.")
        
        # Set PyTorch device
        self.device_torch = torch.device("cuda" if (torch.cuda.is_available() and self.device == 'gpu') else "cpu")
    
    def save(self, out_dir: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Only serializable fields (those with init=True) are saved.
        Computed fields like log_func, base_fn, and device_torch are excluded.
        
        Args:
            out_dir: Output directory path where config.yaml will be saved
        """
        import logging
        
        logger = logging.getLogger(__name__)
        out_dir = Path(out_dir)
        out_p = out_dir / 'config.yaml'
        
        # Only save fields that are part of __init__ (exclude computed fields)
        config_dict = self.as_dict()
        
        with open(out_p, 'w') as f:
            yaml.safe_dump(config_dict, f)
        
        logger.info(f"Saved run config to {out_p}")

    def as_dict(self) -> dict:
        """
        Convert the ProtriderConfig dataclass to a dictionary,
        excluding non-serializable fields.
        
        Returns:
            dict: Dictionary representation of the configuration
        """
        import dataclasses
        
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.init  # Only include fields that are initialized (excludes computed fields)
        }


def load_config(config_path: Union[str, Path]) -> ProtriderConfig:
    """
    Load PROTRIDER configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        ProtriderConfig object with validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError(f"Empty configuration file: {config_path}")
    
    config_dict = _normalize_config_dict(config_dict)
    
    # Convert to ProtriderConfig, which will validate the fields
    try:
        config = ProtriderConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    return config


def _coerce_int(value, field_name: str) -> int:
    """Coerce YAML scalars like 30.0 or \"30\" to int for range() and dataclass fields."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be a whole number, got {value}")
    if isinstance(value, str):
        as_float = float(value)
        if as_float.is_integer():
            return int(as_float)
        raise ValueError(f"{field_name} must be a whole number, got {value!r}")
    raise TypeError(f"{field_name} must be int-like, got {type(value).__name__}")


def _coerce_find_q_method(value) -> str:
    """Normalize fixed latent dimensions to the string form used by the pipeline."""
    if isinstance(value, bool):
        raise ValueError("find_q_method must be 'OHT', 'gs', 'bs' or an integer")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        raise ValueError("find_q_method must be 'OHT', 'gs', 'bs' or an integer")
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"OHT", "gs", "bs"} or normalized.isdigit():
            return normalized
    raise ValueError("find_q_method must be 'OHT', 'gs', 'bs' or an integer")


def _normalize_config_dict(config_dict: dict) -> dict:
    """Normalize types after yaml.safe_load (strings, floats-as-ints, yes/no booleans)."""
    # YAML may load scientific notation as strings depending on the parser/version.
    _float_keys = (
        'lr',
        'inj_freq',
        'inj_mean',
        'inj_sd',
        'min_delta',
        'lambda_presence_absence',
        'pseudocount',
        'outlier_threshold',
        'max_allowed_NAs_per_protein',
        'cohort_stability_drop_fraction',
        'cohort_stability_max_runtime_min',
        'z_threshold',
    )
    for key in _float_keys:
        if key in config_dict and isinstance(config_dict[key], str):
            config_dict[key] = float(config_dict[key])

    _int_keys = (
        "seed",
        "gs_epochs",
        "n_layers",
        "n_epochs",
        "patience",
        "batch_size",
        "h_dim",
        "n_jobs",
        "cohort_stability_n_runs",
        "cohort_stability_min_runs",
        "cohort_stability_min_samples",
        "cohort_stability_seed",
        "cooutlier_min_anomalies",
        "cooutlier_max_clusters",
        "cooutlier_min_samples_for_clustering",
    )
    for key in _int_keys:
        if key in config_dict and config_dict[key] is not None:
            config_dict[key] = _coerce_int(config_dict[key], key)

    if "find_q_method" in config_dict and config_dict["find_q_method"] is not None:
        config_dict["find_q_method"] = _coerce_find_q_method(
            config_dict["find_q_method"]
        )

    _bool_keys = (
        "autoencoder_training",
        "init_pca",
        "presence_absence",
        "common_degrees_freedom",
        "report_all",
        "verbose",
        "use_wandb",
        "cohort_stability",
        "cohort_stability_require_oht",
        "cohort_stability_save_iteration_files",
        "export_latent_space",
        "export_patient_similarity",
        "export_cooutlier_patient_similarity",
    )
    for key in _bool_keys:
        if key in config_dict and config_dict[key] is not None:
            config_dict[key] = _coerce_bool(config_dict[key], key)

    return config_dict


def _coerce_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}")
