import pytest

from silver_torch import compile_silver, parse_silver


PROGRAM = """
pipeline ieee_inverse:
  features voltage, current
  label fault
  architecture transformer
  scaling standard
  missing median
  batch_size 8
  num_workers 2
"""


def test_parse_and_plan_are_inspectable():
    spec = parse_silver(PROGRAM)
    assert spec.name == "ieee_inverse"
    assert spec.features == ("voltage", "current")
    pipeline = compile_silver(PROGRAM).fit(
        [
            {"voltage": 1, "current": 10, "fault": 0},
            {"voltage": 3, "current": 30, "fault": 1},
        ]
    )
    plan = pipeline.plan(device="cuda").to_dict()
    assert plan["schema"] == "silver.torch/plan-2"
    assert plan["input_shape"] == [1, 2]
    assert plan["loader_options"]["pin_memory"] is True
    assert plan["loader_options"]["prefetch_factor"] == 2


def test_fit_uses_training_statistics_and_handles_median():
    pipeline = compile_silver(
        PROGRAM.replace("architecture transformer", "architecture mlp")
    ).fit(
        [
            {"voltage": 1, "current": 10, "fault": 0},
            {"voltage": None, "current": 20, "fault": 1},
            {"voltage": 3, "current": 30, "fault": 1},
        ]
    )
    assert pipeline.statistics["voltage"]["location"] == 2
    assert pipeline.statistics["current"]["location"] == 20


def test_tensors_are_contiguous_when_torch_is_available():
    torch = pytest.importorskip("torch")
    pipeline = compile_silver(PROGRAM).fit(
        [
            {"voltage": 1, "current": 10, "fault": 0},
            {"voltage": 3, "current": 30, "fault": 1},
        ]
    )
    features, labels = pipeline.tensors(
        [{"voltage": 2, "current": 20, "fault": 1}]
    )
    assert features.is_contiguous()
    assert features.dtype == torch.float32
    assert tuple(features.shape) == (1, 1, 2)
    assert tuple(labels.shape) == (1,)


def test_missing_error_is_not_silently_imputed():
    program = PROGRAM.replace("missing median", "missing error")
    pipeline = compile_silver(program).fit(
        [{"voltage": 1, "current": 10, "fault": 0}]
    )
    with pytest.raises(ValueError, match="missing value"):
        pipeline.tensors([{"voltage": None, "current": 10, "fault": 0}])


def test_categorical_features_and_classification_labels_are_encoded():
    program = """
pipeline categorical:
  features sensor, voltage
  categorical sensor
  label fault
  scaling none
  missing unknown
"""
    torch = pytest.importorskip("torch")
    pipeline = compile_silver(program).fit([
        {"sensor": "a", "voltage": 1, "fault": "healthy"},
        {"sensor": "b", "voltage": 2, "fault": "fault"},
    ])
    features, labels = pipeline.transform([
        {"sensor": "unseen", "voltage": 3, "fault": "fault"}
    ])
    assert features[0, 0].item() == 0.0
    assert labels.dtype == torch.long


def test_regression_and_sequence_layout():
    program = """
pipeline sequence:
  features x1, x2, x3, x4
  label target
  architecture rnn
  label_type regression
  sequence_length 2
"""
    torch = pytest.importorskip("torch")
    pipeline = compile_silver(program).fit([
        {"x1": 1, "x2": 2, "x3": 3, "x4": 4, "target": 0.5}
    ])
    features, labels = pipeline.transform([
        {"x1": 1, "x2": 2, "x3": 3, "x4": 4, "target": 0.5}
    ])
    assert tuple(features.shape) == (1, 2, 2)
    assert labels.dtype == torch.float32


def test_invalid_sequence_layout_fails_before_materialization():
    program = PROGRAM.replace("architecture transformer", "architecture transformer").replace(
        "features voltage, current", "features voltage, current, phase"
    ) + "  sequence_length 2\n"
    pipeline = compile_silver(program).fit([
        {"voltage": 1, "current": 10, "phase": 2, "fault": 0},
        {"voltage": 3, "current": 30, "phase": 4, "fault": 1},
    ])
    with pytest.raises(ValueError, match="feature count"):
        pipeline.plan()


def test_cache_and_benchmark(tmp_path):
    pytest.importorskip("torch")
    program = PROGRAM.replace("num_workers 2", "num_workers 0") + "  cache_dir %s\n" % tmp_path
    pipeline = compile_silver(program).fit([
        {"voltage": 1, "current": 10, "fault": 0},
        {"voltage": 3, "current": 30, "fault": 1},
    ])
    rows = [{"voltage": 2, "current": 20, "fault": 1}]
    first = pipeline.dataloader(rows)
    second = pipeline.dataloader(rows)
    assert len(list(first)) == len(list(second)) == 1
    result = pipeline.benchmark(rows, steps=1)
    assert result["samples"] == 1.0
    assert result["samples_per_second"] > 0
