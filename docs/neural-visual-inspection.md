# Inspect every PyTorch layer

`inspect_model()` attaches temporary forward hooks to leaf modules and combines
one observed pass with the model's current parameter gradients.

```python
from silver_torch import inspect_model

sample, _ = next(iter(validation_loader))
inspection = inspect_model(model, sample)
open("network.svg", "w", encoding="utf-8").write(inspection.to_svg())
```

For every leaf layer, the result reports input/output shapes, total and trainable
parameters, activation mean, activation standard deviation, zero fraction, and
gradient RMS. Samples are moved to the model's CPU, CUDA, or MPS device. Hooks
are removed immediately and the prior train/eval state is restored. Statistics
describe the supplied batch and current gradients—not an unsupported global
claim. JSON uses `silver.torch/model-inspection-1`.
