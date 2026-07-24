import numpy as np

from src.train_dp_fedavg import train_dp_fedavg


def test_dp_fedavg_composes_across_rounds():
    rng = np.random.default_rng(42)
    clients = []
    for _ in range(2):
        X = rng.normal(size=(32, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        clients.append((X, y))
    X_test = rng.normal(size=(16, 3)).astype(np.float32)
    y_test = (X_test[:, 0] > 0).astype(np.float32)
    result = train_dp_fedavg(
        clients,
        X_test,
        y_test,
        input_dim=3,
        global_rounds=2,
        local_epochs=1,
        batch_size=16,
        noise_multiplier=1.5,
        seed=3,
    )
    assert result["epsilon_history"][1] > result["epsilon_history"][0]
    assert result["privacy_unit"] == "record"
    assert result["privacy_claim_valid"] is False
    assert result["adjacency"] == "fixed_size_replace_one"


def test_dp_fedavg_target_epsilon_is_respected():
    rng = np.random.default_rng(7)
    clients = []
    for _ in range(2):
        X = rng.normal(size=(24, 2)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        clients.append((X, y))
    X_test = rng.normal(size=(12, 2)).astype(np.float32)
    y_test = (X_test[:, 0] > 0).astype(np.float32)
    result = train_dp_fedavg(
        clients,
        X_test,
        y_test,
        input_dim=2,
        global_rounds=2,
        batch_size=12,
        target_epsilon=2.0,
        seed=7,
    )
    assert result["epsilon"] <= 2.0 + 1e-6
    assert result["privacy_mode"] == "target_epsilon"
