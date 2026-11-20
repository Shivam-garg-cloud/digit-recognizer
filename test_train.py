"""Simple automated tests. Run with:  pytest"""
from sklearn.metrics import accuracy_score

from train import get_models, load_data


def test_dataset_shape():
    (X_train, X_test, y_train, y_test), _ = load_data()
    assert X_train.shape[1] == 64          # 8x8 pixels
    assert len(X_train) + len(X_test) == 1797


def test_every_model_beats_80_percent():
    (X_train, X_test, y_train, y_test), _ = load_data()
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.80, f"{name} scored only {acc:.2%}"
