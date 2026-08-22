import pytest

from silver_torch import ModelBlueprint, TransformerConfig, build_model


def test_transformer_blueprint_builds_optional_pytorch_model():
    torch = pytest.importorskip("torch")
    model = build_model(ModelBlueprint("tiny", "transformer", {
        "vocab_size": 32, "hidden_size": 16, "layers": 1, "heads": 4,
        "max_sequence_length": 8, "output_size": 3,
    }))
    output = model(torch.randint(0, 32, (2, 8)))
    assert tuple(output.shape) == (2, 3)


def test_transformer_config_rejects_invalid_attention_shape():
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(vocab_size=10, hidden_size=10, heads=3)


@pytest.mark.parametrize("field,value", [
    ("output_size", 0),
    ("max_sequence_length", 0),
    ("feedforward_size", 0),
    ("dropout", 1.0),
])
def test_transformer_config_rejects_invalid_runtime_dimensions(field, value):
    with pytest.raises(ValueError):
        TransformerConfig(vocab_size=10, **{field: value})


def test_transformer_rejects_sequences_longer_than_configured_maximum():
    torch = pytest.importorskip("torch")
    model = build_model(ModelBlueprint("tiny", "transformer", {
        "vocab_size": 16, "hidden_size": 8, "layers": 1, "heads": 2,
        "max_sequence_length": 2,
    }))
    with pytest.raises(ValueError, match="max_sequence_length"):
        model(torch.randint(0, 16, (1, 3)))
