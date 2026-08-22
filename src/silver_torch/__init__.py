"""Silver's optional, inspectable PyTorch preprocessing layer."""

from .pipeline import SilverTorchPipeline, compile_silver, parse_silver
from .spec import SilverPreprocessSpec
from .modeling import ModelBlueprint, TransformerConfig, build_model, build_transformer
from .trainer import (
    EpochMetrics,
    SilverTrainer,
    TrainingConfig,
    TrainingResult,
    build_tabular_model,
    count_parameters,
)

__all__ = ["SilverPreprocessSpec", "SilverTorchPipeline", "compile_silver", "parse_silver",
           "ModelBlueprint", "TransformerConfig", "build_model", "build_transformer"]
__all__ += [
    "EpochMetrics", "SilverTrainer", "TrainingConfig", "TrainingResult",
    "build_tabular_model", "count_parameters",
]

__version__ = "1.0.0"
