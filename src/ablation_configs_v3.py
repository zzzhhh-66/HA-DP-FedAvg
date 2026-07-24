"""Factorial ablation protocol for HA-DP-FedAvg v3."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationSpec:
    """One cell in the 2 x 2 HA-weighting/privacy-schedule experiment."""

    ablation_id: str
    paper_label: str
    description: str
    use_heterogeneity_aware: bool
    epsilon_strategy: str


FACTORIAL_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec(
        ablation_id="A0",
        paper_label="DP-FedAvg (matched)",
        description="No HA weighting + uniform privacy schedule",
        use_heterogeneity_aware=False,
        epsilon_strategy="uniform",
    ),
    AblationSpec(
        ablation_id="A1",
        paper_label="+ Privacy Scheduler",
        description="No HA weighting + back-loaded privacy schedule",
        use_heterogeneity_aware=False,
        epsilon_strategy="back_loaded",
    ),
    AblationSpec(
        ablation_id="A2",
        paper_label="+ HA Weighting",
        description="HA weighting + uniform privacy schedule",
        use_heterogeneity_aware=True,
        epsilon_strategy="uniform",
    ),
    AblationSpec(
        ablation_id="A3",
        paper_label="Full HA-DP-FedAvg",
        description="HA weighting + back-loaded privacy schedule",
        use_heterogeneity_aware=True,
        epsilon_strategy="back_loaded",
    ),
)


def validate_factorial_ablation_specs(
    specs: tuple[AblationSpec, ...] = FACTORIAL_ABLATIONS,
) -> None:
    """Require exactly one experiment for every binary factor combination."""
    if len({spec.ablation_id for spec in specs}) != len(specs):
        raise ValueError("Ablation IDs must be unique.")

    expected = {
        (False, "uniform"),
        (False, "back_loaded"),
        (True, "uniform"),
        (True, "back_loaded"),
    }
    actual = {
        (spec.use_heterogeneity_aware, spec.epsilon_strategy)
        for spec in specs
    }
    if actual != expected:
        raise ValueError(
            "The ablation protocol must cover the complete 2 x 2 factorial design."
        )

