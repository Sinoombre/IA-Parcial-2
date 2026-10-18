"""The model feature set and forbidden observation fields."""

from .settings import agent_config

ALLOWED_PREDICTORS = tuple(agent_config()["features"]["allowed_predictors"])
FORBIDDEN_AGENT_FIELDS = {
    "Churn Value",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "would_churn",
    "saved",
    "q_i",
    "r_true",
}


def validate_observation(columns) -> None:
    forbidden = sorted(set(columns) & FORBIDDEN_AGENT_FIELDS)
    if forbidden:
        raise ValueError(f"La percepción contiene campos prohibidos: {forbidden}")
