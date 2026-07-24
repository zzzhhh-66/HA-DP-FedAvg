import numpy as np


def create_iid_clients(
    X_train,
    y_train,
    num_clients: int,
    random_state: int = 42,
):
    """Randomly split training data into IID clients."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if num_clients > len(y_train):
        raise ValueError("num_clients cannot exceed the number of training records.")
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y_train))
    rng.shuffle(indices)

    client_indices = np.array_split(indices, num_clients)
    clients = []

    for idx in client_indices:
        clients.append((X_train[idx], y_train[idx]))

    return clients


def create_noniid_clients_dirichlet(
    X_train,
    y_train,
    num_clients: int,
    alpha: float = 0.5,
    random_state: int = 42,
    min_size: int = 10,
    max_attempts: int = 1000,
):
    """Create a minimum-size-conditioned Dirichlet Non-IID client split.

    Draws that leave any client below ``min_size`` are rejected, so this is a
    constrained/truncated protocol rather than an unconditional Dirichlet draw.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if min_size <= 0:
        raise ValueError("min_size must be positive.")
    if num_clients * min_size > len(y_train):
        raise ValueError("num_clients * min_size exceeds the training-set size.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    rng = np.random.default_rng(random_state)
    y_train_int = y_train.astype(int)
    class_ids = np.unique(y_train_int)

    for _ in range(max_attempts):
        client_indices = [[] for _ in range(num_clients)]

        for class_id in class_ids:
            class_indices = np.where(y_train_int == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            class_splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        if min(len(idx) for idx in client_indices) >= min_size:
            break
    else:
        raise RuntimeError(
            "Could not construct a Dirichlet split satisfying min_size; "
            "increase alpha, lower min_size, or increase max_attempts."
        )

    clients = []
    for idx in client_indices:
        idx = np.array(idx)
        rng.shuffle(idx)
        clients.append((X_train[idx], y_train[idx]))

    return clients


def print_client_distribution(clients, name: str = "Clients") -> None:
    print(f"===== {name} =====")
    for i, (_, y_client) in enumerate(clients):
        print(f"Client {i + 1}: n={len(y_client)}, positive_ratio={y_client.mean():.4f}")
    print()


def create_profile_matched_test_clients(
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_clients: list[tuple[np.ndarray, np.ndarray]],
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create held-out label-skew slices aligned with training-client profiles.

    For each class, held-out records are allocated in proportion to the class
    counts observed in the corresponding training clients. Features are not
    used for allocation. The slices therefore measure robustness to the
    simulated label-skew profiles without evaluating on local training data.
    """
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same length.")
    if not train_clients:
        raise ValueError("At least one training client is required.")

    rng = np.random.default_rng(random_state)
    y_test_int = np.asarray(y_test).astype(int)
    assignments: list[list[int]] = [[] for _ in train_clients]

    for class_id in np.unique(y_test_int):
        indices = np.flatnonzero(y_test_int == class_id)
        rng.shuffle(indices)
        train_counts = np.asarray(
            [np.sum(np.asarray(client_y).astype(int) == class_id) for _, client_y in train_clients],
            dtype=float,
        )
        if train_counts.sum() <= 0:
            proportions = np.full(len(train_clients), 1.0 / len(train_clients))
        else:
            proportions = train_counts / train_counts.sum()

        expected = proportions * len(indices)
        counts = np.floor(expected).astype(int)
        remainder = len(indices) - int(counts.sum())
        if remainder:
            order = np.argsort(-(expected - counts), kind="stable")
            counts[order[:remainder]] += 1

        start = 0
        for client_id, count in enumerate(counts):
            stop = start + int(count)
            assignments[client_id].extend(indices[start:stop].tolist())
            start = stop

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for indices in assignments:
        selected = np.asarray(indices, dtype=int)
        rng.shuffle(selected)
        result.append((X_test[selected], y_test[selected]))
    return result
