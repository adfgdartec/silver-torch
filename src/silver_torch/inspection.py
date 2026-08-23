"""Real PyTorch model inspection and deterministic neural-network SVGs."""

from dataclasses import asdict, dataclass
from html import escape
import math
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class LayerInspection:
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: int
    trainable_parameters: int
    activation_mean: Optional[float] = None
    activation_std: Optional[float] = None
    activation_zero_fraction: Optional[float] = None
    gradient_rms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["input_shape"] = list(self.input_shape)
        value["output_shape"] = list(self.output_shape)
        return value


@dataclass(frozen=True)
class ModelInspection:
    model_type: str
    layers: Tuple[LayerInspection, ...]
    parameters: int
    trainable_parameters: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "silver.torch/model-inspection-1",
            "model_type": self.model_type,
            "parameters": self.parameters,
            "trainable_parameters": self.trainable_parameters,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def to_svg(self, *, title: str = "Neural network inspection") -> str:
        return neural_network_svg(self, title=title)


def inspect_model(model: Any, example_input: Optional[Any] = None) -> ModelInspection:
    """Inspect leaf modules, shapes, activations, parameters, and current gradients."""
    torch = _torch()
    modules = [(name or "model", module) for name, module in model.named_modules()
               if not any(module.children())]
    observed: Dict[str, Dict[str, Any]] = {}
    handles = []

    def capture(name: str):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            input_tensor = _first_tensor(torch, inputs)
            output_tensor = _first_tensor(torch, output)
            value: Dict[str, Any] = {
                "input_shape": _shape(input_tensor),
                "output_shape": _shape(output_tensor),
            }
            if output_tensor is not None and output_tensor.numel():
                numeric = output_tensor.detach().float()
                value.update({
                    "activation_mean": float(numeric.mean().item()),
                    "activation_std": float(numeric.std(unbiased=False).item()),
                    "activation_zero_fraction": float((numeric == 0).float().mean().item()),
                })
            observed[name] = value
        return hook

    if example_input is not None:
        for name, module in modules:
            handles.append(module.register_forward_hook(capture(name)))
        was_training = bool(model.training)
        try:
            model.eval()
            with torch.no_grad():
                prepared = _move_to_model_device(torch, model, example_input)
                if isinstance(prepared, dict):
                    model(**prepared)
                elif isinstance(prepared, tuple):
                    model(*prepared)
                else:
                    model(prepared)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

    layers = []
    for name, module in modules:
        parameters = tuple(module.parameters(recurse=False))
        gradients = [parameter.grad.detach().float().reshape(-1) for parameter in parameters
                     if parameter.grad is not None and parameter.grad.numel()]
        gradient_rms = None
        if gradients:
            joined = torch.cat(gradients)
            gradient_rms = float(torch.sqrt(torch.mean(joined * joined)).item())
        state = observed.get(name, {})
        layers.append(LayerInspection(
            name=name,
            layer_type=type(module).__name__,
            input_shape=state.get("input_shape", ()),
            output_shape=state.get("output_shape", ()),
            parameters=sum(int(parameter.numel()) for parameter in parameters),
            trainable_parameters=sum(int(parameter.numel()) for parameter in parameters
                                     if parameter.requires_grad),
            activation_mean=state.get("activation_mean"),
            activation_std=state.get("activation_std"),
            activation_zero_fraction=state.get("activation_zero_fraction"),
            gradient_rms=gradient_rms,
        ))
    all_parameters = tuple(model.parameters())
    return ModelInspection(
        model_type=type(model).__name__,
        layers=tuple(layers),
        parameters=sum(int(parameter.numel()) for parameter in all_parameters),
        trainable_parameters=sum(int(parameter.numel()) for parameter in all_parameters
                                 if parameter.requires_grad),
    )


def neural_network_svg(inspection: ModelInspection, *, title: str = "Neural network inspection") -> str:
    """Render an accessible, self-contained SVG from an actual model inspection."""
    width = 1080
    card_height = 94
    height = 154 + max(1, len(inspection.layers)) * card_height
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{escape(title)}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif}"
        ".title{font-size:26px;font-weight:700;fill:#f5f7fa}.meta{font-size:13px;fill:#9fb0c3}"
        ".name{font-size:15px;font-weight:700;fill:#e8eef7}.small{font-size:12px;fill:#aebed0}"
        ".value{font-size:12px;font-weight:600;fill:#7de3ff}</style>",
        f'<rect width="{width}" height="{height}" rx="22" fill="#0b1017"/>',
        '<rect x="20" y="20" width="1040" height="86" rx="16" fill="#111a25" stroke="#294155"/>',
        f'<text class="title" x="42" y="55">{escape(title)}</text>',
        f'<text class="meta" x="42" y="82">{escape(inspection.model_type)} · '
        f'{inspection.parameters:,} parameters · {inspection.trainable_parameters:,} trainable</text>',
    ]
    if not inspection.layers:
        parts.append('<text class="meta" x="42" y="140">No leaf layers discovered.</text>')
    for index, layer in enumerate(inspection.layers):
        y = 126 + index * card_height
        accent = "#42d6ff" if index % 2 == 0 else "#9b7cff"
        parts.extend([
            f'<rect x="30" y="{y}" width="1020" height="76" rx="13" fill="#101923" stroke="#26394b"/>',
            f'<rect x="30" y="{y}" width="6" height="76" rx="3" fill="{accent}"/>',
            f'<text class="name" x="54" y="{y + 27}">{escape(layer.name)}</text>',
            f'<text class="small" x="54" y="{y + 51}">{escape(layer.layer_type)}</text>',
            f'<text class="small" x="260" y="{y + 27}">shape</text>',
            f'<text class="value" x="260" y="{y + 51}">{escape(_shape_text(layer.input_shape))} → {escape(_shape_text(layer.output_shape))}</text>',
            f'<text class="small" x="530" y="{y + 27}">parameters</text>',
            f'<text class="value" x="530" y="{y + 51}">{layer.parameters:,}</text>',
            f'<text class="small" x="700" y="{y + 27}">activation μ / σ / zero</text>',
            f'<text class="value" x="700" y="{y + 51}">{_number(layer.activation_mean)} / {_number(layer.activation_std)} / {_percent(layer.activation_zero_fraction)}</text>',
            f'<text class="small" x="950" y="{y + 27}">grad RMS</text>',
            f'<text class="value" x="950" y="{y + 51}">{_number(layer.gradient_rms)}</text>',
        ])
    parts.append("</svg>")
    return "".join(parts)


def _first_tensor(torch: Any, value: Any) -> Optional[Any]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if torch.is_tensor(item)), None)
    if isinstance(value, dict):
        return next((item for item in value.values() if torch.is_tensor(item)), None)
    return None


def _move_to_model_device(torch: Any, model: Any, value: Any) -> Any:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return value
    device = parameter.device
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_model_device(torch, model, item) for item in value)
    if isinstance(value, list):
        return [_move_to_model_device(torch, model, item) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_model_device(torch, model, item) for key, item in value.items()}
    return value


def _shape(value: Optional[Any]) -> Tuple[int, ...]:
    return tuple(int(item) for item in value.shape) if value is not None else ()


def _shape_text(value: Tuple[int, ...]) -> str:
    return "×".join(str(item) for item in value) if value else "not observed"


def _number(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.3g}"


def _percent(value: Optional[float]) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise ImportError("install silver-torch[pytorch] to inspect models") from error
    return torch
