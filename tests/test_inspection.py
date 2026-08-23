import pytest

from silver_torch import inspect_model, neural_network_svg


def test_inspect_model_captures_real_shapes_activations_gradients_and_svg():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
    )
    sample = torch.tensor([[1.0, -2.0, 0.5], [0.2, 0.4, -0.7]])
    model(sample).sum().backward()

    inspection = inspect_model(model, sample)
    assert inspection.parameters == 26
    assert [layer.layer_type for layer in inspection.layers] == ["Linear", "ReLU", "Linear"]
    assert inspection.layers[0].input_shape == (2, 3)
    assert inspection.layers[-1].output_shape == (2, 2)
    assert inspection.layers[0].gradient_rms is not None
    assert inspection.layers[1].activation_zero_fraction is not None
    assert inspection.to_dict()["schema"] == "silver.torch/model-inspection-1"

    svg = neural_network_svg(inspection, title="Real model")
    assert svg.startswith("<svg")
    assert "Real model" in svg
    assert "activation μ / σ / zero" in svg
    assert "grad RMS" in svg


def test_inspection_without_sample_still_reports_topology():
    torch = pytest.importorskip("torch")
    inspection = inspect_model(torch.nn.Linear(2, 1))
    assert inspection.layers[0].parameters == 3
    assert inspection.layers[0].input_shape == ()
