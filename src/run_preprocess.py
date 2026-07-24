"""Run preprocessing pipeline and client split validation."""

from .preprocess import (
    preprocess_default_credit_card_clients,
    preprocess_german_credit_numeric,
    preprocess_give_me_some_credit,
    print_dataset_summary,
)
from .split_clients import (
    create_iid_clients,
    create_noniid_clients_dirichlet,
    print_client_distribution,
)
from .utils import resolve_data_path


def main():
    gms_path = resolve_data_path("gmsc")
    german_path = resolve_data_path("german")
    default_credit_path = resolve_data_path("default_credit")

    print("=" * 60)
    print("Give Me Some Credit")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_give_me_some_credit(
        str(gms_path)
    )
    print_dataset_summary(X_train, X_val, X_test, y_train, y_val, y_test, "Give Me Some Credit")
    print(f"Feature count: {len(feature_names)}")

    iid_clients = create_iid_clients(X_train, y_train, num_clients=10)
    print_client_distribution(iid_clients, "Give Me Some Credit - IID Clients (10)")

    noniid_clients_05 = create_noniid_clients_dirichlet(X_train, y_train, num_clients=10, alpha=0.5)
    print_client_distribution(noniid_clients_05, "Give Me Some Credit - Non-IID Clients alpha=0.5")

    noniid_clients_01 = create_noniid_clients_dirichlet(X_train, y_train, num_clients=10, alpha=0.1)
    print_client_distribution(noniid_clients_01, "Give Me Some Credit - Non-IID Clients alpha=0.1")


    print("=" * 60)
    print("Default of Credit Card Clients")
    print("=" * 60)

    (
        X_train_dc,
        X_val_dc,
        X_test_dc,
        y_train_dc,
        y_val_dc,
        y_test_dc,
        feature_names_dc,
    ) = preprocess_default_credit_card_clients(str(default_credit_path))
    print_dataset_summary(
        X_train_dc,
        X_val_dc,
        X_test_dc,
        y_train_dc,
        y_val_dc,
        y_test_dc,
        "Default of Credit Card Clients",
    )
    print(f"Feature count: {len(feature_names_dc)}")

    iid_clients_dc = create_iid_clients(X_train_dc, y_train_dc, num_clients=10)
    print_client_distribution(
        iid_clients_dc,
        "Default of Credit Card Clients - IID Clients (10)",
    )

    noniid_clients_dc_05 = create_noniid_clients_dirichlet(
        X_train_dc, y_train_dc, num_clients=10, alpha=0.5
    )
    print_client_distribution(
        noniid_clients_dc_05,
        "Default of Credit Card Clients - Non-IID Clients alpha=0.5",
    )

    noniid_clients_dc_01 = create_noniid_clients_dirichlet(
        X_train_dc, y_train_dc, num_clients=10, alpha=0.1
    )
    print_client_distribution(
        noniid_clients_dc_01,
        "Default of Credit Card Clients - Non-IID Clients alpha=0.1",
    )

    print("=" * 60)
    print("German Credit (numeric)")
    print("=" * 60)

    (
        X_train_de,
        X_val_de,
        X_test_de,
        y_train_de,
        y_val_de,
        y_test_de,
        feature_names_de,
    ) = preprocess_german_credit_numeric(str(german_path))
    print_dataset_summary(
        X_train_de,
        X_val_de,
        X_test_de,
        y_train_de,
        y_val_de,
        y_test_de,
        "German Credit",
    )
    print(f"Feature count: {len(feature_names_de)}")

    iid_clients_de = create_iid_clients(X_train_de, y_train_de, num_clients=5)
    print_client_distribution(iid_clients_de, "German Credit - IID Clients (5)")

    noniid_clients_de = create_noniid_clients_dirichlet(
        X_train_de, y_train_de, num_clients=5, alpha=0.5
    )
    print_client_distribution(noniid_clients_de, "German Credit - Non-IID Clients alpha=0.5")


if __name__ == "__main__":
    main()
