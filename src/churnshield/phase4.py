"""Compatibilidad: importar aquí la API del motor de escenarios de Fase 4."""

from .simulation.phase4 import SCENARIOS, run_many, run_scenario, update_r_hat

__all__ = ["SCENARIOS", "run_many", "run_scenario", "update_r_hat"]
