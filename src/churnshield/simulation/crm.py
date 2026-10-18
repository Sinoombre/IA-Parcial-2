"""CRM simulado: fallos explícitos e idempotencia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CRMResult:
    campaign_id: str
    customer_id: str
    outcome: str
    attempts: int
    executed: bool
    first_outcome: str = ""
    second_outcome: str = ""
    final_status: str = ""
    retry_used: bool = False


class SimulatedCRM:
    def __init__(
        self,
        *,
        timeout_rate: float = 0.0,
        rejected_rate: float = 0.0,
        error_rate: float = 0.0,
        timeout_retries: int = 1,
    ):
        rates = [timeout_rate, rejected_rate, error_rate]
        if any(not 0 <= float(x) <= 1 for x in rates) or sum(rates) > 1:
            raise ValueError(
                "las tasas del CRM deben estar en [0,1] y sumar como máximo 1"
            )
        self.timeout_rate = float(timeout_rate)
        self.rejected_rate = float(rejected_rate)
        self.error_rate = float(error_rate)
        self.timeout_retries = int(timeout_retries)
        if self.timeout_retries < 0:
            raise ValueError("timeout_retries no puede ser negativo")
        self._results: dict[str, CRMResult] = {}

    def _one_attempt(
        self, campaign_id: str, customer_id: str, rng: np.random.Generator
    ) -> str:
        draw = float(rng.random())
        if draw < self.timeout_rate:
            return "timeout"
        if draw < self.timeout_rate + self.rejected_rate:
            return "rejected"
        if draw < self.timeout_rate + self.rejected_rate + self.error_rate:
            return "error"
        return "accepted"

    def send(
        self, campaign_id: str, customer_id: str, rng: np.random.Generator
    ) -> CRMResult:
        """Envía una campaña; repetir el mismo ID nunca crea una segunda acción."""
        if campaign_id in self._results:
            return self._results[campaign_id]
        attempts = 0
        outcome = "timeout"
        first_outcome = ""
        second_outcome = ""
        while attempts <= self.timeout_retries:
            attempts += 1
            outcome = self._one_attempt(campaign_id, customer_id, rng)
            if attempts == 1:
                first_outcome = outcome
            elif attempts == 2:
                second_outcome = outcome
            if outcome != "timeout":
                break
        final_status = "crm_error" if outcome == "timeout" and attempts > 1 else outcome
        result = CRMResult(
            campaign_id=campaign_id,
            customer_id=str(customer_id),
            outcome=outcome,
            attempts=attempts,
            executed=(outcome == "accepted"),
            first_outcome=first_outcome,
            second_outcome=second_outcome,
            final_status=final_status,
            retry_used=(attempts > 1),
        )
        self._results[campaign_id] = result
        return result

    @property
    def results(self) -> tuple[CRMResult, ...]:
        return tuple(self._results.values())
