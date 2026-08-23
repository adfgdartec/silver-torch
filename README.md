# silver-torch

<p align="center"><img src="https://raw.githubusercontent.com/adfgdartec/silver-torch/main/docs/assets/silver-hero.png" alt="Silver PyTorch training" width="100%"></p>

**Leakage-safe tensors and a real training loop with every important decision
left visible.**

[![PyPI](https://img.shields.io/pypi/v/silver-torch?color=7c3aed)](https://pypi.org/project/silver-torch/)
[![CI](https://github.com/adfgdartec/silver-torch/actions/workflows/ci.yml/badge.svg)](https://github.com/adfgdartec/silver-torch/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-c0c0c0)](LICENSE)

An optional PyTorch layer for Silver. It turns a small Silver preprocessing
program into a fitted, inspectable, reusable tensor and `DataLoader` pipeline.

## Every layer, explained

<p align="center"><img src="https://raw.githubusercontent.com/adfgdartec/silver-torch/main/docs/assets/neural-network-inspection.png" alt="Neural-network inputs, hidden layers, activations, gradients, prediction, and training health" width="100%"></p>

```python
from silver_torch import inspect_model

# Use the loader created above (or any validation DataLoader).
sample, _ = next(iter(loader))
inspection = inspect_model(model, sample)
open("network.svg", "w", encoding="utf-8").write(inspection.to_svg())
```

The SVG is generated from the model itself: leaf-layer shapes, parameter counts,
activation mean/variance/sparsity, and current gradient RMS. It works across
CPU, CUDA, and MPS. Read the [measurement details](docs/neural-visual-inspection.md).

```bash
pip install silver-data
pip install 'silver-torch[pytorch]'
```

## Train a real model

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from silver_torch import (
    SilverTrainer, TrainingConfig, build_tabular_model,
)

x = torch.tensor([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
y = torch.tensor([0, 0, 1, 1])
loader = DataLoader(TensorDataset(x, y), batch_size=4)

model = build_tabular_model(input_size=2, output_size=2)
trainer = SilverTrainer(model, TrainingConfig(
    epochs=50,
    learning_rate=0.01,
    checkpoint_path=".silver/best.pt",
))
result = trainer.fit(loader, loader)

print(result.best_epoch, result.best_loss, result.parameter_count)
```

`SilverTrainer` includes deterministic seeds, CPU/CUDA/MPS selection, gradient
clipping, non-finite loss protection, early stopping, best-state restoration,
atomic checkpoints, evaluation, prediction, and event callbacks.

```python
from silver_data import Dataset
from silver_torch import compile_silver

program = """
pipeline ieee_inverse:
  features voltage, current, phase, sensor
  categorical sensor
  label fault
  architecture transformer
  sequence_length 2
  scaling standard
  missing median
  label_type classification
  batch_size 128
  num_workers 2
  cache_dir .cache/ieee_inverse
"""

dataset = Dataset.from_records("ieee", [
    {"voltage": 1.0, "current": 2.0, "phase": 0.2, "sensor": "a", "fault": 0},
    {"voltage": 1.2, "current": 2.1, "phase": 0.3, "sensor": "b", "fault": 1},
])
splits = dataset.split(0.5, 0.5, 0.0)
pipeline = compile_silver(program).fit(splits.train.records())
loader = pipeline.dataloader(splits.validation.records(), device="cuda")
print(pipeline.plan(device="cuda").to_dict())
print(pipeline.benchmark(splits.validation.records(), steps=20))
```

The compiler has explicit research-safety boundaries:

- statistics and vocabularies are fitted only on `splits.train`;
- missing columns, non-finite numbers, invalid labels, and incompatible
  sequence lengths fail loudly;
- categorical vocabularies are sorted for reproducibility and reserve index 0
  for unknown values;
- classification targets are `torch.long`; regression targets use the chosen
  floating dtype;
- cache keys include the fitted-training fingerprint and transformed rows, and
  cache writes are atomic;
- loaders are seeded and tune pinning, persistent workers, prefetching, and
  `drop_last` based on the declared runtime.

The emitted shapes are `[batch, features]` for MLP, `[batch, 1, features]` for
CNN, and `[batch, sequence_length, features_per_step]` for RNN/Transformer.
These are layout contracts, not model implementations. Measure with
`benchmark()` on the target machine; input speedups depend on storage, CPU,
worker count, batch size, and accelerator.
