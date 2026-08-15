"""Declarative model builders with an optional PyTorch execution backend."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    hidden_size: int = 256
    layers: int = 4
    heads: int = 8
    feedforward_size: Optional[int] = None
    max_sequence_length: int = 2048
    output_size: int = 2
    task: str = "classification"
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 1 or self.hidden_size < 1 or self.layers < 1 or self.heads < 1:
            raise ValueError("transformer dimensions must be positive")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if self.task not in ("classification", "regression", "language_modeling"):
            raise ValueError("unsupported transformer task")


@dataclass(frozen=True)
class ModelBlueprint:
    name: str
    architecture: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "architecture": self.architecture, "config": dict(self.config)}


def build_transformer(config: TransformerConfig) -> Any:
    """Build a standard encoder transformer when PyTorch is installed."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise ImportError("install silver-torch[pytorch] to build models") from error

    class SilverTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position = nn.Embedding(config.max_sequence_length, config.hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.heads,
                dim_feedforward=config.feedforward_size or config.hidden_size * 4,
                dropout=config.dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.head = nn.Linear(config.hidden_size, config.output_size)

        def forward(self, tokens: Any) -> Any:
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            encoded = self.embedding(tokens) + self.position(positions)
            mask = None
            if config.task == "language_modeling":
                mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device), diagonal=1).bool()
            hidden = self.encoder(encoded, mask=mask)
            if config.task == "language_modeling":
                return self.head(hidden)
            return self.head(hidden[:, 0, :])

    return SilverTransformer()


def build_model(blueprint: ModelBlueprint) -> Any:
    if blueprint.architecture == "transformer":
        return build_transformer(TransformerConfig(**blueprint.config))
    raise ValueError("no built-in builder for architecture %r; register an adapter for custom models" % blueprint.architecture)
