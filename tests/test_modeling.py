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
