"""Compile Silver preprocessing intent into a leakage-safe PyTorch pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .spec import SilverPreprocessSpec


def parse_silver(source: str) -> SilverPreprocessSpec:
    """Parse the data-pipeline subset of Silver with strict field handling."""
    values: Dict[str, Any] = {}
    supported = {"features", "categorical", "label", "architecture", "scaling", "missing",
                 "label_type", "dtype", "batch_size", "shuffle", "num_workers",
                 "prefetch_factor", "drop_last", "sequence_length", "seed", "cache_dir"}
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pipeline "):
            match = re.fullmatch(r"pipeline\s+([A-Za-z_]\w*)\s*:", line)
            if not match:
                raise ValueError("pipeline declaration must look like 'pipeline name:'")
            values["name"] = match.group(1)
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError("Silver pipeline fields must look like 'field value'")
        key, raw_value = parts
        if key not in supported:
            raise ValueError("unsupported Silver preprocessing field: %s" % key)
        raw_value = raw_value.strip().strip('"')
        if key in ("features", "categorical"):
            values[key] = tuple(x.strip() for x in raw_value.split(",") if x.strip())
        elif raw_value.lower() in ("true", "false"):
            values[key] = raw_value.lower() == "true"
        elif re.fullmatch(r"\d+", raw_value):
            values[key] = int(raw_value)
        else:
            values[key] = raw_value
    values.setdefault("name", "silver_pipeline")
    if "features" not in values or "label" not in values:
        raise ValueError("Silver pipeline requires 'features' and 'label'")
    return SilverPreprocessSpec(**values)


@dataclass(frozen=True)
class ColumnState:
    kind: str
    location: float = 0.0
    scale: float = 1.0
    categories: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledPlan:
    spec: SilverPreprocessSpec
    feature_count: int
    input_shape: Tuple[int, ...]
    loader_options: Dict[str, Any]
    fitted_rows: int
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": "silver.torch/plan-2", "name": self.spec.name,
                "architecture": self.spec.architecture, "feature_count": self.feature_count,
                "input_shape": list(self.input_shape), "label_type": self.spec.label_type,
                "loader_options": dict(self.loader_options), "fitted_rows": self.fitted_rows,
                "fingerprint": self.fingerprint}


class SilverTorchPipeline:
    """Fit transforms on training rows, then reuse them without data leakage."""

    def __init__(self, spec: SilverPreprocessSpec):
        self.spec = spec
        self._states: Dict[str, ColumnState] = {}
        self._label_mapping: Dict[str, int] = {}
        self._fitted_rows = 0
        self._fingerprint = ""

    @property
    def fitted(self) -> bool:
        return bool(self._states)

    @property
    def statistics(self) -> Dict[str, Dict[str, Any]]:
        """Return fitted state without exposing mutable internal objects."""
        self._require_fit()
        return {name: {"kind": state.kind, "location": state.location,
                       "scale": state.scale, "categories": list(state.categories)}
                for name, state in self._states.items()}

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> "SilverTorchPipeline":
        materialized = list(rows)
        if not materialized:
            raise ValueError("cannot fit an empty dataset")
        missing_columns = [c for c in self.spec.features + (self.spec.label,)
                           if c not in materialized[0]]
        if missing_columns:
            raise ValueError("missing dataset columns: %s" % ", ".join(missing_columns))
        states: Dict[str, ColumnState] = {}
        for column in self.spec.features:
            if column in self.spec.categorical:
                categories = sorted({self._as_category(row.get(column)) for row in materialized
                                     if row.get(column) not in (None, "")})
                states[column] = ColumnState("categorical", categories=tuple(categories))
                continue
            numbers = [self._number(row.get(column), column) for row in materialized]
            observed = [v for v in numbers if v is not None]
            if not observed and self.spec.missing == "error":
                raise ValueError("feature %r has no observed values" % column)
            location = self._impute_location(observed)
            clean = [v if v is not None else location for v in numbers]
            if self.spec.scaling == "standard":
                location = sum(clean) / len(clean)
                scale = math.sqrt(sum((v - location) ** 2 for v in clean) / len(clean)) or 1.0
            elif self.spec.scaling == "minmax":
                location = min(clean)
                scale = (max(clean) - location) or 1.0
            else:
                scale = 1.0
            states[column] = ColumnState("numeric", location, scale)
        if self.spec.label_type == "classification":
            labels = sorted({self._as_category(row.get(self.spec.label)) for row in materialized})
            self._label_mapping = {value: index for index, value in enumerate(labels)}
        self._states = states
        self._fitted_rows = len(materialized)
        self._fingerprint = self._hash_rows(materialized)
        return self

    def plan(self, device: str = "cpu") -> CompiledPlan:
        self._require_fit()
        feature_count = len(self.spec.features)
        if self.spec.architecture == "mlp":
            shape = (feature_count,)
        elif self.spec.architecture == "cnn":
            shape = (1, feature_count)
        else:
            if feature_count % self.spec.sequence_length:
                raise ValueError("feature count must divide sequence_length for sequential models")
            shape = (self.spec.sequence_length, feature_count // self.spec.sequence_length)
        options: Dict[str, Any] = {"batch_size": self.spec.batch_size, "shuffle": self.spec.shuffle,
                                    "num_workers": self.spec.num_workers,
                                    "pin_memory": device.startswith("cuda"),
                                    "persistent_workers": self.spec.num_workers > 0,
                                    "drop_last": self.spec.drop_last}
        if self.spec.num_workers > 0:
            options["prefetch_factor"] = self.spec.prefetch_factor
        return CompiledPlan(self.spec, feature_count, shape, options, self._fitted_rows, self._fingerprint)

    def transform(self, rows: Iterable[Mapping[str, Any]]) -> Tuple[Any, Any]:
        """Materialize a feature tensor and correctly typed target tensor."""
        self._require_fit()
        torch = _torch()
        features: List[List[float]] = []
        labels: List[Any] = []
        for row in rows:
            features.append([self._transform_value(row.get(column), column) for column in self.spec.features])
            labels.append(self._transform_label(row.get(self.spec.label)))
        x = torch.tensor(features, dtype=getattr(torch, self.spec.dtype)).contiguous()
        label_dtype = torch.long if self.spec.label_type == "classification" else getattr(torch, self.spec.dtype)
        y = torch.tensor(labels, dtype=label_dtype).contiguous()
        if self.spec.architecture == "cnn":
            x = x.unsqueeze(1)
        elif self.spec.architecture in ("rnn", "transformer"):
            x = x.reshape(len(features), self.spec.sequence_length, -1)
        return x, y

    def tensors(self, rows: Iterable[Mapping[str, Any]]) -> Tuple[Any, Any]:
        """Backward-compatible alias for :meth:`transform`."""
        return self.transform(rows)

    def save(self, path: str) -> None:
        """Persist the fitted preprocessing contract without requiring PyTorch."""
        self._require_fit()
        payload = {
            "schema": "silver.torch/fitted-pipeline-1",
            "spec": asdict(self.spec),
            "states": {name: asdict(state) for name, state in self._states.items()},
            "label_mapping": dict(self._label_mapping),
            "fitted_rows": self._fitted_rows,
            "fingerprint": self._fingerprint,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "SilverTorchPipeline":
        """Restore a fitted pipeline for inference or serving."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "silver.torch/fitted-pipeline-1":
            raise ValueError("unsupported Silver Torch pipeline schema")
        spec_values = payload["spec"]
        spec_values["features"] = tuple(spec_values["features"])
        spec_values["categorical"] = tuple(spec_values["categorical"])
        pipeline = cls(SilverPreprocessSpec(**spec_values))
        pipeline._states = {
            name: ColumnState(**state) for name, state in payload["states"].items()
        }
        pipeline._label_mapping = dict(payload.get("label_mapping", {}))
        pipeline._fitted_rows = int(payload["fitted_rows"])
        pipeline._fingerprint = str(payload["fingerprint"])
        return pipeline

    def dataloader(self, rows: Iterable[Mapping[str, Any]], device: str = "cpu") -> Any:
        torch = _torch()
        from torch.utils.data import DataLoader, TensorDataset
        x, y = self._cached_or_transform(rows)
        options = self.plan(device).loader_options
        generator = torch.Generator()
        generator.manual_seed(self.spec.seed)
        options["generator"] = generator
        return DataLoader(TensorDataset(x, y), **options)

    def benchmark(self, rows: Iterable[Mapping[str, Any]], steps: int = 50,
                  device: str = "cpu") -> Dict[str, float]:
        """Measure input throughput, including host-to-device transfer when requested."""
        if steps < 1:
            raise ValueError("steps must be positive")
        loader = self.dataloader(rows, device=device)
        start = time.perf_counter()
        seen = 0
        for index, (features, labels) in enumerate(loader):
            if device.startswith("cuda"):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                _torch().cuda.synchronize()
            seen += len(labels)
            if index + 1 >= steps:
                break
        elapsed = max(time.perf_counter() - start, 1e-9)
        return {"samples": float(seen), "seconds": elapsed, "samples_per_second": seen / elapsed}

    def _cached_or_transform(self, rows: Iterable[Mapping[str, Any]]) -> Tuple[Any, Any]:
        materialized = list(rows)
        if not self.spec.cache_dir:
            return self.transform(materialized)
        torch = _torch()
        cache_root = Path(self.spec.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256((self._fingerprint + json.dumps(materialized, sort_keys=True, default=str)).encode()).hexdigest()[:24]
        path = cache_root / (self.spec.name + "-" + key + ".pt")
        if path.exists():
            payload = torch.load(str(path), map_location="cpu")
            return payload["features"], payload["labels"]
        x, y = self.transform(materialized)
        fd, temp_name = tempfile.mkstemp(prefix="silver-", suffix=".pt", dir=str(cache_root))
        os.close(fd)
        try:
            torch.save({"features": x, "labels": y}, temp_name)
            os.replace(temp_name, str(path))
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return x, y

    def _transform_value(self, value: Any, column: str) -> float:
        state = self._states[column]
        if state.kind == "categorical":
            category = self._as_category(value)
            return float(state.categories.index(category) + 1) if category in state.categories else 0.0
        number = self._number(value, column)
        if number is None:
            if self.spec.missing == "error":
                raise ValueError("missing value in feature %r" % column)
            number = state.location
        return (number - state.location) / state.scale

    def _transform_label(self, value: Any) -> Any:
        if self.spec.label_type == "classification":
            category = self._as_category(value)
            if category not in self._label_mapping:
                raise ValueError("unknown label %r after fitting" % value)
            return self._label_mapping[category]
        number = self._number(value, self.spec.label)
        if number is None:
            raise ValueError("regression label cannot be missing")
        return number

    def _impute_location(self, observed: Sequence[float]) -> float:
        if self.spec.missing in ("mean", "unknown") and observed:
            return sum(observed) / len(observed)
        if self.spec.missing == "median" and observed:
            values = sorted(observed)
            middle = len(values) // 2
            return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        return 0.0

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("fit the pipeline on the training split before transform")

    @staticmethod
    def _as_category(value: Any) -> str:
        return "<missing>" if value in (None, "") else str(value)

    @staticmethod
    def _number(value: Any, column: str) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("feature %r must be numeric" % column) from error
        if not math.isfinite(result):
            raise ValueError("feature %r must be finite" % column)
        return result

    @staticmethod
    def _hash_rows(rows: Sequence[Mapping[str, Any]]) -> str:
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def compile_silver(source: str) -> SilverTorchPipeline:
    return SilverTorchPipeline(parse_silver(source))


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise ImportError("install silver-torch[pytorch] to materialize PyTorch tensors") from error
    return torch
