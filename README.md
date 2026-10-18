# IA-Parcial-2
Desarrollo del segundo parcial de la materia de inteligencia artificial a cargo del profesor Edwin Puertas

## Automatizaciones de GitHub

- **Evaluación manual:** desde Actions, ejecuta `Manual scenario evaluation`;
  elige un escenario o todos, una semilla y si quieres correr las cinco semillas
  predefinidas. Los resultados generados se publican como artefacto durante 14
  días.
- **Calidad:** `Code quality` ejecuta Ruff para revisar errores de lint en el
  código y formato en los módulos de simulación y escenarios.
- **Actualizaciones:** Dependabot revisa semanalmente las dependencias Python y
  las acciones de GitHub.

Para ejecutar un escenario manualmente desde PowerShell, configura el código
fuente en la ruta de importación y corre el CLI:

```powershell
$env:PYTHONPATH = "src"
python -m churnshield scenarios --scenario S02 --seed 101
```

Los logs operativos se escriben en `artifacts/scenarios/` y la verdad del
evaluador, por separado, en `artifacts/evaluator_truth/`. Se puede cambiar el
destino con `--evaluator-output-dir`; debe ser una ruta separada del directorio
de resultados operativos. Para la simulación de cohortes, `simulate` ofrece la
misma opción.
