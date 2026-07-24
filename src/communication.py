def model_size_bytes(num_parameters: int, bytes_per_param: int = 4) -> int:
    return num_parameters * bytes_per_param


def communication_per_round(num_parameters: int, num_clients: int, bytes_per_param: int = 4) -> int:
    """Server broadcast + client upload in one round."""
    model_bytes = model_size_bytes(num_parameters, bytes_per_param)
    return 2 * num_clients * model_bytes


def total_communication_cost(
    num_parameters: int,
    num_clients: int,
    num_rounds: int,
    bytes_per_param: int = 4,
) -> tuple[int, int]:
    per_round = communication_per_round(num_parameters, num_clients, bytes_per_param)
    total = per_round * num_rounds
    return per_round, total


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)
