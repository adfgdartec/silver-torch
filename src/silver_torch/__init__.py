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
from .inspection import LayerInspection, ModelInspection, inspect_model, neural_network_svg

__all__ = ["SilverPreprocessSpec", "SilverTorchPipeline", "compile_silver", "parse_silver",
           "ModelBlueprint", "TransformerConfig", "build_model", "build_transformer"]
__all__ += [
    "EpochMetrics", "SilverTrainer", "TrainingConfig", "TrainingResult",
    "build_tabular_model", "count_parameters",
    "LayerInspection", "ModelInspection", "inspect_model", "neural_network_svg",
]

__version__ = "1.5.1"
