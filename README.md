# silver-torch

An optional PyTorch layer for Silver. It turns a small Silver preprocessing
program into a fitted, inspectable, reusable tensor and `DataLoader` pipeline.

```bash
pip install silver-data
pip install 'silver-torch[pytorch]'
```

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
    {"voltage": 1.0, "current": 2.0, "phase": 0.2, "fault": 0},
    {"voltage": 1.2, "current": 2.1, "phase": 0.3, "fault": 1},
])
splits = dataset.split(0.8, 0.1, 0.1)
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
