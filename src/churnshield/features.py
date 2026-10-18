"""Features permitidas y columnas prohibidas para el agente."""
ALLOWED_PREDICTORS = ["Partner", "Dependents", "Phone Service", "Multiple Lines", "Internet Service", "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing", "Payment Method", "Tenure Months", "Monthly Charges", "Total Charges"]
FORBIDDEN_AGENT_FIELDS = {"Churn Value", "Churn Label", "Churn Score", "Churn Reason", "would_churn", "saved", "q_i", "r_true"}


def validate_observation(columns) -> None:
    forbidden = sorted(set(columns) & FORBIDDEN_AGENT_FIELDS)
    if forbidden:
        raise ValueError(f"La percepción contiene campos prohibidos: {forbidden}")
