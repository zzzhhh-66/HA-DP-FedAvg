import numpy as np
import pytest

from src.split_clients import (
    create_iid_clients,
    create_noniid_clients_dirichlet,
    create_profile_matched_test_clients,
)


def test_iid_split_is_complete_and_reproducible():
    X = np.arange(200, dtype=np.float32).reshape(100, 2)
    y = np.array([0, 1] * 50, dtype=np.float32)
    clients_a = create_iid_clients(X, y, num_clients=5, random_state=11)
    clients_b = create_iid_clients(X, y, num_clients=5, random_state=11)
    assert sum(len(labels) for _, labels in clients_a) == len(y)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(clients_a, clients_b, strict=True))


def test_noniid_split_rejects_impossible_minimum():
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.array([0, 1] * 5, dtype=np.float32)
    with pytest.raises(ValueError):
        create_noniid_clients_dirichlet(X, y, num_clients=3, min_size=4)


def test_noniid_split_supports_non_contiguous_class_labels():
    X = np.arange(120, dtype=np.float32).reshape(60, 2)
    y = np.array([1, 3] * 30, dtype=np.float32)
    clients = create_noniid_clients_dirichlet(
        X, y, num_clients=3, alpha=1.0, min_size=2, random_state=5
    )
    assert sum(len(labels) for _, labels in clients) == len(y)
    assert set(np.concatenate([labels for _, labels in clients])) == {1.0, 3.0}


def test_profile_matched_test_slices_are_complete_and_disjoint():
    X_test = np.arange(80, dtype=np.float32).reshape(40, 2)
    y_test = np.array([0] * 30 + [1] * 10, dtype=np.float32)
    train_clients = [
        (np.zeros((10, 2)), np.array([0] * 9 + [1], dtype=np.float32)),
        (np.zeros((10, 2)), np.array([0] * 4 + [1] * 6, dtype=np.float32)),
    ]
    slices = create_profile_matched_test_clients(
        X_test,
        y_test,
        train_clients,
        random_state=3,
    )
    recovered_rows = np.concatenate([features[:, 0] for features, _ in slices])
    assert len(recovered_rows) == len(X_test)
    assert len(np.unique(recovered_rows)) == len(X_test)
    assert sum(len(labels) for _, labels in slices) == len(y_test)
    assert np.mean(slices[0][1]) < np.mean(slices[1][1])
