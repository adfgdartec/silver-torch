"""Small, production-minded PyTorch training loops with inspectable results."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = 1.0
    task: str = "classification"
    device: str = "auto"
    early_stopping_patience: Optional[int] = 5
    min_delta: float = 0.0
    seed: int = 17
    checkpoint_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("epochs and learning rate must be positive; decay cannot be negative")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive when provided")
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive when provided")
        if self.min_delta < 0 or self.seed < 0:
            raise ValueError("min_delta and seed must be non-negative")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: Optional[float]
    train_metric: Optional[float]
    validation_metric: Optional[float]
    gradient_norm: float
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    history: Tuple[EpochMetrics, ...]
    best_epoch: int
    best_loss: float
    stopped_early: bool
    device: str
    parameter_count: int
    duration: float

    def metric_history(self) -> Tuple[Dict[str, float], ...]:
        values = []
        for epoch in self.history:
            metrics = {"loss": epoch.train_loss, "gradient_norm": epoch.gradient_norm}
            if epoch.validation_loss is not None:
                metrics["val_loss"] = epoch.validation_loss
            if epoch.train_metric is not None:
                metrics["metric"] = epoch.train_metric
            if epoch.validation_metric is not None:
                metrics["val_metric"] = epoch.validation_metric
            values.append(metrics)
        return tuple(values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "silver.torch/training-result-1",
            "history": [epoch.to_dict() for epoch in self.history],
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "stopped_early": self.stopped_early,
            "device": self.device,
            "parameter_count": self.parameter_count,
            "duration": self.duration,
        }


class SilverTrainer:
    """Train ordinary PyTorch modules with deterministic, inspectable defaults."""

    def __init__(self, model: Any, config: Optional[TrainingConfig] = None):
        self.model = model
        self.config = config or TrainingConfig()
        self._device = "cpu"

    @property
    def device(self) -> str:
        return self._device

    def fit(
        self,
        train_loader: Iterable[Any],
        validation_loader: Optional[Iterable[Any]] = None,
        *,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Callable[[Any, Any], Any]] = None,
        on_epoch: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> TrainingResult:
        torch = _torch()
        started = time.perf_counter()
        _seed(torch, self.config.seed)
        self._device = _resolve_device(torch, self.config.device)
        self.model.to(self._device)
        optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = loss_fn or _default_loss(torch, self.config.task)
        history: List[EpochMetrics] = []
        best_loss = math.inf
        best_epoch = 0
        best_state = None
        stale_epochs = 0
        stopped_early = False
        for epoch in range(1, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss, train_metric, gradient_norm = self._train_epoch(
                train_loader, optimizer, loss_fn
            )
            validation_loss = validation_metric = None
            if validation_loader is not None:
                validation_loss, validation_metric = self.evaluate(
                    validation_loader, loss_fn=loss_fn
                )
            monitored = validation_loss if validation_loss is not None else train_loss
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                train_metric=train_metric,
                validation_metric=validation_metric,
                gradient_norm=gradient_norm,
                duration=max(0.0, time.perf_counter() - epoch_started),
            )
            history.append(metrics)
            if monitored < best_loss - self.config.min_delta:
                best_loss = monitored
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                if self.config.checkpoint_path:
                    _save_checkpoint(
                        torch, self.config.checkpoint_path, self.model,
                        optimizer, self.config, metrics,
                    )
            else:
                stale_epochs += 1
            if on_epoch is not None:
                callback_value = on_epoch({
                    "kind": "epoch",
                    "epoch": epoch,
                    "metrics": _flat_metrics(metrics),
                })
                if callback_value is not None:
                    raise TypeError("on_epoch callbacks must return None")
            patience = self.config.early_stopping_patience
            if patience is not None and stale_epochs >= patience:
                stopped_early = True
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self._device)
        return TrainingResult(
            history=tuple(history),
            best_epoch=best_epoch,
            best_loss=best_loss,
            stopped_early=stopped_early,
            device=self._device,
            parameter_count=count_parameters(self.model),
            duration=max(0.0, time.perf_counter() - started),
        )

    def evaluate(
        self,
        loader: Iterable[Any],
        *,
        loss_fn: Optional[Callable[[Any, Any], Any]] = None,
    ) -> Tuple[float, Optional[float]]:
        torch = _torch()
        loss_fn = loss_fn or _default_loss(torch, self.config.task)
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        metric_total = 0.0
        with torch.no_grad():
            for batch in loader:
                features, labels = _batch(batch)
                features = features.to(self._device)
                labels = labels.to(self._device)
                outputs = _outputs(self.model(features))
                labels = _labels_for_task(labels, outputs, self.config.task)
                loss = loss_fn(outputs, labels)
                size = int(labels.shape[0])
                total_loss += float(loss.detach().item()) * size
                total_examples += size
                metric_total += _metric(torch, outputs, labels, self.config.task) * size
        if total_examples == 0:
            raise ValueError("cannot evaluate an empty loader")
        return total_loss / total_examples, metric_total / total_examples

    def predict(self, loader: Iterable[Any]) -> Any:
        torch = _torch()
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                features = _prediction_features(batch).to(self._device)
                outputs = _outputs(self.model(features))
                if self.config.task == "classification":
                    outputs = outputs.argmax(dim=-1)
                predictions.append(outputs.detach().cpu())
        if not predictions:
            raise ValueError("cannot predict from an empty loader")
        return torch.cat(predictions, dim=0)

    def _train_epoch(
        self, loader: Iterable[Any], optimizer: Any, loss_fn: Callable[[Any, Any], Any]
    ) -> Tuple[float, Optional[float], float]:
        torch = _torch()
        self.model.train()
        total_loss = 0.0
        total_examples = 0
        metric_total = 0.0
        gradient_norm = 0.0
        for batch in loader:
            features, labels = _batch(batch)
            features = features.to(self._device)
            labels = labels.to(self._device)
            optimizer.zero_grad(set_to_none=True)
            outputs = _outputs(self.model(features))
            labels = _labels_for_task(labels, outputs, self.config.task)
            loss = loss_fn(outputs, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            loss.backward()
            if self.config.gradient_clip is not None:
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            else:
                norm = _gradient_norm(torch, self.model.parameters())
            gradient_norm = max(gradient_norm, float(norm))
            optimizer.step()
            size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * size
            total_examples += size
            metric_total += _metric(torch, outputs.detach(), labels, self.config.task) * size
        if total_examples == 0:
            raise ValueError("cannot train on an empty loader")
        return total_loss / total_examples, metric_total / total_examples, gradient_norm


def build_tabular_model(
    input_size: int,
    output_size: int,
    *,
    hidden_sizes: Sequence[int] = (128, 64),
    dropout: float = 0.1,
) -> Any:
    """Build a strong default MLP for numerical/categorical tabular tensors."""
    if input_size < 1 or output_size < 1 or any(size < 1 for size in hidden_sizes):
        raise ValueError("model dimensions must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    try:
        from torch import nn
    except ImportError as error:
        raise ImportError("install silver-torch[pytorch] to build PyTorch models") from error

    layers: List[Any] = []
    previous = input_size
    for size in hidden_sizes:
        layers.extend([nn.Linear(previous, size), nn.ReLU(), nn.LayerNorm(size)])
        if dropout:
            layers.append(nn.Dropout(dropout))
        previous = size
    layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


def count_parameters(model: Any, *, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not getattr(torch.backends, "mps", None):
            raise RuntimeError("MPS was requested but is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_loss(torch: Any, task: str) -> Any:
    return torch.nn.CrossEntropyLoss() if task == "classification" else torch.nn.MSELoss()


def _batch(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, dict):
        features = batch.get("features", batch.get("inputs"))
        labels = batch.get("labels", batch.get("targets"))
        if features is None or labels is None:
            raise ValueError("mapping batches require features/inputs and labels/targets")
        return features, labels
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("batches must be a (features, labels) pair or mapping")


def _prediction_features(batch: Any) -> Any:
    if isinstance(batch, dict):
        value = batch.get("features", batch.get("inputs"))
        if value is None:
            raise ValueError("mapping batches require features or inputs")
        return value
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _outputs(value: Any) -> Any:
    return value.logits if hasattr(value, "logits") else value


def _labels_for_task(labels: Any, outputs: Any, task: str) -> Any:
    if task == "classification":
        return labels.long().reshape(-1)
    return labels.to(dtype=outputs.dtype).reshape_as(outputs)


def _metric(torch: Any, outputs: Any, labels: Any, task: str) -> float:
    if task == "classification":
        return float((outputs.argmax(dim=-1) == labels).float().mean().item())
    return float(torch.sqrt(torch.mean((outputs - labels) ** 2)).item())


def _gradient_norm(torch: Any, parameters: Iterable[Any]) -> float:
    squares = [parameter.grad.detach().norm(2).pow(2)
               for parameter in parameters if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0


def _flat_metrics(metrics: EpochMetrics) -> Dict[str, float]:
    values = {
        "loss": metrics.train_loss,
        "gradient_norm": metrics.gradient_norm,
    }
    if metrics.validation_loss is not None:
        values["val_loss"] = metrics.validation_loss
    if metrics.train_metric is not None:
        values["metric"] = metrics.train_metric
    if metrics.validation_metric is not None:
        values["val_metric"] = metrics.validation_metric
    return values


def _save_checkpoint(
    torch: Any,
    path: str,
    model: Any,
    optimizer: Any,
    config: TrainingConfig,
    metrics: EpochMetrics,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".silver-", suffix=".pt", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        torch.save({
            "schema": "silver.torch/checkpoint-1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": metrics.to_dict(),
        }, temp_name)
        os.replace(temp_name, str(destination))
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise ImportError("install silver-torch[pytorch] to train PyTorch models") from error
    return torch
