"""Ablation study configurations for HA-DP-FedAvg."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    ablation_id: str
    description: str
    method: str
    split: str = "noniid"
    alpha: float | None = 0.5
    noise_multiplier: float = 1.0
    use_heterogeneity_aware: bool = False
    privacy_mode: str = "epsilon_allocation"
    total_epsilon: float = 3.0
    epsilon_strategy: str = "back_loaded"
    noise_schedule: str = "decreasing"
    metadata_epsilon: float = 0.0
    dp_target_epsilon: float | None = None
    balance_gamma: float = 1.0
    ha_strength: float = 0.5
    ha_prior_strength: float = 20.0


GMSC_ABLATION_EXPERIMENTS: list[AblationConfig] = [
    AblationConfig(
        ablation_id="B1",
        description="DP-FedAvg baseline: sample-size weights + fixed noise",
        method="dp_fedavg",
        use_heterogeneity_aware=False,
        noise_multiplier=1.0,
    ),
    AblationConfig(
        ablation_id="B2",
        description="Privacy-matched DP-FedAvg: sample weights + uniform composed budget",
        method="dp_fedavg",
        use_heterogeneity_aware=False,
        dp_target_epsilon=3.0,
    ),
    AblationConfig(
        ablation_id="P1",
        description="Full HA-DP-FedAvg: HA weights + back-loaded RDP cost schedule",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=3.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P2",
        description="HA weights + uniform epsilon allocation (ablation on privacy schedule)",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="uniform",
        total_epsilon=3.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P3",
        description="Back-loaded schedule only (ablation: w/o HA aggregation)",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=False,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=3.0,
    ),
    AblationConfig(
        ablation_id="P4",
        description="HA aggregation only (ablation: w/o epsilon allocation)",
        method="dp_fedavg",
        use_heterogeneity_aware=True,
        noise_multiplier=1.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P5",
        description="HA weights + front-loaded RDP cost schedule",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="front_loaded",
        total_epsilon=3.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P6",
        description="Original hard HA weights (ablation: w/o bounded shrinkage)",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=3.0,
        metadata_epsilon=0.1,
        ha_strength=1.0,
        ha_prior_strength=0.0,
    ),
]

GERMAN_ABLATION_EXPERIMENTS: list[AblationConfig] = [
    AblationConfig(
        ablation_id="B1",
        description="DP-FedAvg baseline: sample-size weights + fixed noise",
        method="dp_fedavg",
        use_heterogeneity_aware=False,
        noise_multiplier=0.7,
    ),
    AblationConfig(
        ablation_id="B2",
        description="Privacy-matched DP-FedAvg: sample weights + uniform composed budget",
        method="dp_fedavg",
        use_heterogeneity_aware=False,
        dp_target_epsilon=2.0,
    ),
    AblationConfig(
        ablation_id="P1",
        description="Full HA-DP-FedAvg: HA weights + back-loaded RDP cost schedule",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=2.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P2",
        description="HA weights + uniform epsilon allocation",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="uniform",
        total_epsilon=2.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P3",
        description="Back-loaded schedule only (w/o HA aggregation)",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=False,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=2.0,
    ),
    AblationConfig(
        ablation_id="P4",
        description="HA aggregation only (w/o epsilon allocation)",
        method="dp_fedavg",
        use_heterogeneity_aware=True,
        noise_multiplier=0.7,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P5",
        description="HA weights + front-loaded RDP cost schedule",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="front_loaded",
        total_epsilon=2.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P6",
        description="Original hard HA weights (ablation: w/o bounded shrinkage)",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=2.0,
        metadata_epsilon=0.1,
        ha_strength=1.0,
        ha_prior_strength=0.0,
    ),
]


DEFAULT_CREDIT_ABLATION_EXPERIMENTS: list[AblationConfig] = [
    AblationConfig(
        ablation_id="B1",
        description="DP-FedAvg baseline: sample-size weights + fixed noise",
        method="dp_fedavg",
        use_heterogeneity_aware=False,
        noise_multiplier=1.0,
    ),
    AblationConfig(
        ablation_id="B2",
        description="Privacy-matched DP-FedAvg: sample weights + uniform composed budget",
        method="dp_fedavg",
        use_heterogeneity_aware=False,
        dp_target_epsilon=3.0,
    ),
    AblationConfig(
        ablation_id="P1",
        description="Full HA-DP-FedAvg: HA weights + back-loaded RDP cost schedule",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=3.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P2",
        description="HA weights + uniform epsilon allocation",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="uniform",
        total_epsilon=3.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P3",
        description="Back-loaded schedule only (w/o HA aggregation)",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=False,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=3.0,
    ),
    AblationConfig(
        ablation_id="P4",
        description="HA aggregation only (w/o epsilon allocation)",
        method="dp_fedavg",
        use_heterogeneity_aware=True,
        noise_multiplier=1.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P5",
        description="HA weights + front-loaded RDP cost schedule",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="front_loaded",
        total_epsilon=3.0,
        metadata_epsilon=0.1,
    ),
    AblationConfig(
        ablation_id="P6",
        description="Original hard HA weights (ablation: w/o bounded shrinkage)",
        method="ha_dp_fedavg",
        use_heterogeneity_aware=True,
        privacy_mode="epsilon_allocation",
        epsilon_strategy="back_loaded",
        total_epsilon=3.0,
        metadata_epsilon=0.1,
        ha_strength=1.0,
        ha_prior_strength=0.0,
    ),
]


def get_ablation_experiments(dataset: str) -> list[AblationConfig]:
    if dataset == "gmsc":
        return list(GMSC_ABLATION_EXPERIMENTS)
    if dataset == "german":
        return list(GERMAN_ABLATION_EXPERIMENTS)
    if dataset == "default_credit":
        return list(DEFAULT_CREDIT_ABLATION_EXPERIMENTS)
    raise ValueError(f"Unknown dataset for ablation: {dataset}")


def validate_ablation_config(config: AblationConfig) -> None:
    valid_methods = {"dp_fedavg", "ha_dp_fedavg"}
    if config.method not in valid_methods:
        raise ValueError(f"Invalid ablation method: {config.method}")

    if config.split == "noniid" and config.alpha is None:
        raise ValueError(f"Ablation {config.ablation_id} requires alpha for non-IID split.")

    if (
        config.method == "dp_fedavg"
        and config.dp_target_epsilon is None
        and config.noise_multiplier <= 0
    ):
        raise ValueError(f"Ablation {config.ablation_id} requires positive noise_multiplier.")
    if config.dp_target_epsilon is not None and config.dp_target_epsilon <= 0:
        raise ValueError(f"Ablation {config.ablation_id} has invalid dp_target_epsilon.")
    if config.balance_gamma < 0:
        raise ValueError(f"Ablation {config.ablation_id} has invalid balance_gamma.")
    if not 0 <= config.ha_strength <= 1:
        raise ValueError(f"Ablation {config.ablation_id} has invalid ha_strength.")
    if config.ha_prior_strength < 0:
        raise ValueError(f"Ablation {config.ablation_id} has invalid ha_prior_strength.")

    if config.method == "ha_dp_fedavg":
        if config.privacy_mode not in {"epsilon_allocation", "noise_schedule"}:
            raise ValueError(f"Invalid privacy_mode: {config.privacy_mode}")
        if config.privacy_mode == "epsilon_allocation" and config.total_epsilon <= 0:
            raise ValueError(f"Ablation {config.ablation_id} requires positive total_epsilon.")
        if config.metadata_epsilon < 0 or config.metadata_epsilon >= config.total_epsilon:
            raise ValueError(f"Ablation {config.ablation_id} has an invalid metadata_epsilon.")
