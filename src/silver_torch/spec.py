"""Validated, serialisable configuration for Silver's PyTorch compiler."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class SilverPreprocessSpec:
    name: str
    features: Tuple[str, ...]
    label: str
    architecture: str = "mlp"
    categorical: Tuple[str, ...] = ()
    scaling: str = "standard"
    missing: str = "error"
    label_type: str = "classification"
    dtype: str = "float32"
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int = 2
    drop_last: bool = False
    sequence_length: int = 1
    seed: int = 17
    cache_dir: str = ""
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.features or not self.label:
            raise ValueError("name, features, and label are required")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must not contain duplicates")
        unknown = set(self.categorical) - set(self.features)
        if unknown:
            raise ValueError("categorical columns must be listed in features: %s" % sorted(unknown))
        if self.architecture not in ("mlp", "cnn", "rnn", "transformer"):
            raise ValueError("architecture must be mlp, cnn, rnn, or transformer")
        if self.scaling not in ("none", "standard", "minmax"):
            raise ValueError("scaling must be none, standard, or minmax")
        if self.missing not in ("error", "zero", "mean", "median", "unknown"):
            raise ValueError("missing must be error, zero, mean, median, or unknown")
        if self.label_type not in ("classification", "regression"):
            raise ValueError("label_type must be classification or regression")
        if self.dtype not in ("float16", "float32", "float64", "bfloat16"):
            raise ValueError("dtype must be a floating PyTorch dtype")
        if self.batch_size < 1 or self.num_workers < 0 or self.prefetch_factor < 1:
            raise ValueError("batch_size, workers, and prefetch_factor are invalid")
        if self.sequence_length < 1 or self.seed < 0:
            raise ValueError("sequence_length must be positive and seed non-negative")

