import pytest

from silver_torch import (
    SilverTrainer,
    TrainingConfig,
    build_tabular_model,
    count_parameters,
)


def _classification_loaders():
    torch = pytest.importorskip("torch")
    features = torch.tensor([
        [-2.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-0.5, -1.0],
        [1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0],
    ])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)


def test_trainer_fits_restores_best_state_and_emits_events(tmp_path):
    pytest.importorskip("torch")
    loader = _classification_loaders()
    model = build_tabular_model(2, 2, hidden_sizes=(8,), dropout=0)
    events = []
    trainer = SilverTrainer(model, TrainingConfig(
        epochs=20,
        learning_rate=0.05,
        early_stopping_patience=5,
        checkpoint_path=str(tmp_path / "best.pt"),
        seed=3,
    ))
    result = trainer.fit(loader, loader, on_epoch=events.append)

    assert result.best_epoch >= 1
    assert result.best_loss < 0.2
    assert result.parameter_count == count_parameters(model)
    assert len(events) == len(result.history)
    assert events[0]["kind"] == "epoch"
    assert (tmp_path / "best.pt").exists()
    assert result.to_dict()["schema"] == "silver.torch/training-result-1"
    assert result.metric_history()[0]["loss"] >= 0
    loss, accuracy = trainer.evaluate(loader)
    assert loss < 0.2
    assert accuracy == 1.0
    assert trainer.predict(loader).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_regression_trainer_supports_mapping_batches():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1)
    batches = [{
        "features": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "targets": torch.tensor([[1.0], [3.0], [5.0], [7.0]]),
    }]
    trainer = SilverTrainer(model, TrainingConfig(
        epochs=100, learning_rate=0.05, task="regression",
        early_stopping_patience=None, gradient_clip=None,
    ))
    result = trainer.fit(batches)
    assert result.best_loss < 0.05
    loss, rmse = trainer.evaluate(batches)
    assert loss < 0.05
    assert rmse < 0.25


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"learning_rate": 0}, {"gradient_clip": 0},
    {"task": "ranking"}, {"early_stopping_patience": 0},
])
def test_training_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_tabular_model_validates_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        build_tabular_model(0, 2)
    with pytest.raises(ValueError, match="dropout"):
        build_tabular_model(2, 2, dropout=1)
