"""Silver's optional, inspectable PyTorch preprocessing layer."""

from .pipeline import SilverTorchPipeline, compile_silver, parse_silver
from .spec import SilverPreprocessSpec

__all__ = ["SilverPreprocessSpec", "SilverTorchPipeline", "compile_silver", "parse_silver"]

__version__ = "0.2.0"
