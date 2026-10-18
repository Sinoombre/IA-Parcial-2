"""Compatibilidad: importar aquí el CRM simulado del subpaquete simulation."""

from .simulation.crm import CRMResult, SimulatedCRM

__all__ = ["CRMResult", "SimulatedCRM"]
