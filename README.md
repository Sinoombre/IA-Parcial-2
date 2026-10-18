# IA-Parcial-2
Desarrollo del segundo parcial de la materia de inteligencia artificial a cargo del profesor Edwin Puertas

## Preparar y activar el proyecto

El proyecto está probado con **Python 3.12**. Los siguientes comandos son para
PowerShell desde la carpeta raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH = "src"
```

Si PowerShell bloquea la activación del entorno virtual, permite scripts solo
en la ventana actual y vuelve a activarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Para salir del entorno virtual:

```powershell
deactivate
```

## Comandos principales

Ejecuta estos comandos desde la raíz, con el entorno virtual activo y
`PYTHONPATH` configurado como arriba.

```powershell
# Ejecutar todas las pruebas
python -m pytest -q

# Revisar lint y formato con Ruff
python -m ruff check src tests experiments ui
python -m ruff format --check `
  src/churnshield/scenarios src/churnshield/simulation `
  tests/scenarios tests/simulation ui `
  src/churnshield/evaluation.py src/churnshield/final_eval.py `
  src/churnshield/robustness.py src/churnshield/data.py `
  src/churnshield/s03.py tests/evaluation tests/final `
  tests/agent/test_config.py tests/modeling

# Entrenar y evaluar el modelo usando solo datos de desarrollo
python -m churnshield dev-eval

# Ejecutar la prueba básica del agente con datos sintéticos
python -m churnshield run --scenario S03
python -m churnshield run --scenario S03 --json

# Iniciar la interfaz local en http://127.0.0.1:5000
python -m ui.app
```

La interfaz carga el modelo y su tarjeta desde `artifacts/dev_eval/`. Ejecuta
`dev-eval` antes de iniciar la interfaz si todavía no existen esos artefactos.

La evaluación final reservada **no forma parte de las comprobaciones rutinarias**.
Solo se ejecuta con autorización explícita después de congelar el protocolo:

```powershell
python -m churnshield evaluate-final --help
# Ejecutar únicamente si la pareja autorizó abrir la reserva:
python -m churnshield evaluate-final --confirm-final
```

## Automatizaciones de GitHub

- **Evaluación manual:** desde Actions, ejecuta `Manual scenario evaluation`;
  elige un escenario o todos, una semilla y si quieres correr las cinco semillas
  predefinidas. Los resultados generados se publican como artefacto durante 14
  días.
- **Calidad:** `Code quality` ejecuta Ruff para revisar errores de lint en el
  código y formato en los módulos de simulación y escenarios.
- **Actualizaciones:** Dependabot revisa semanalmente las dependencias Python y
  las acciones de GitHub.

Para ejecutar escenarios manualmente:

```powershell
python -m churnshield scenarios --scenario S02 --seed 101
python -m churnshield scenarios --all --all-seeds
python -m churnshield evaluate --input-dir artifacts/scenarios --output artifacts/scenario_evaluation.json
python -m churnshield simulate --seed 101
```

Los logs operativos se escriben en `artifacts/scenarios/` y la verdad del
evaluador, por separado, en `artifacts/evaluator_truth/`. Se puede cambiar el
destino con `--evaluator-output-dir`; debe ser una ruta separada del directorio
de resultados operativos. Para la simulación de cohortes, `simulate` ofrece la
misma opción. Consulta la ayuda de cualquier comando con:

```powershell
python -m churnshield --help
python -m churnshield scenarios --help
python -m churnshield evaluate --help
python -m churnshield simulate --help
```
