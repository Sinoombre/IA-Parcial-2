# IA-Parcial-2

Proyecto de predicción y simulación de retención para el segundo parcial de
Inteligencia Artificial.

## Preparar el entorno

El proyecto se prueba con Python 3.12. Desde PowerShell, en la raíz del repo:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH = "src"
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Configuración del modelo

`config/agente.yaml` es la fuente canónica de predictores. El modelo, la
evaluación y la interfaz usan exactamente sus nueve campos `allowed_predictors`;
las ocho columnas bajo `excluded_by_design` no se usan como features. El
model card y el manifiesto registran ese mismo conjunto y los hashes del
dataset, la partición, la configuración y los umbrales.

La interfaz usa un modelo de desarrollo de `artifacts/dev_eval/`. Antes de
predecir verifica el model card; si el modelo falta, la tarjeta es inválida o
las features no coinciden, responde con HTTP 503 en vez de inventar valores.
Para regenerar el modelo, ejecuta `dev-eval` después de integrar el código:

```powershell
python -m churnshield dev-eval
```

## Pruebas y calidad

```powershell
python -m pytest -q
python -m ruff check src tests experiments ui
python -m ruff format --check `
  src/churnshield/scenarios src/churnshield/simulation `
  tests/scenarios tests/simulation tests/modeling tests/ui ui `
  src/churnshield/evaluation.py src/churnshield/final_eval.py `
  src/churnshield/robustness.py src/churnshield/data.py `
  src/churnshield/settings.py src/churnshield/features.py `
  src/churnshield/model.py src/churnshield/dev_eval.py `
  src/churnshield/protocol.py src/churnshield/__main__.py `
  src/churnshield/perception.py tests/evaluation tests/final
```

El workflow **Code quality** ejecuta Ruff y toda la suite de pytest.

## Escenarios sintéticos

Los escenarios S02–S10 usan fixtures sintéticos y no cargan la población de
clientes. Los comandos escriben resultados en `artifacts/scenarios/` y la
verdad del evaluador, por separado, en `artifacts/evaluator_truth/`. Para
ejecutar todos los escenarios y cinco semillas sin mezclar ejecuciones
anteriores, crea directorios únicos:

```powershell
$runId = [guid]::NewGuid().ToString("N")
$scenarioDir = Join-Path "artifacts\runs" "$runId\scenarios"
$truthDir = Join-Path "artifacts\runs" "$runId\evaluator_truth"
$evaluationFile = Join-Path "artifacts\runs" "$runId\evaluation.json"
New-Item -ItemType Directory -Path $scenarioDir, $truthDir | Out-Null

python -m churnshield scenarios --all --all-seeds `
  --output-dir $scenarioDir `
  --evaluator-output-dir $truthDir
if ($LASTEXITCODE -ne 0) { throw "Falló la ejecución de escenarios" }

python -m churnshield evaluate `
  --input-dir $scenarioDir `
  --output $evaluationFile `
  --expected-scenarios S02 S03 S04 S05 S06 S07 S08 S09 S10 `
  --expected-seeds 101 102 103 104 105
```

El evaluador rechaza carpetas sin ejecuciones y puede exigir el conjunto
exacto escenario/semilla esperado. Cada invocación del workflow manual usa un
directorio propio bajo `artifacts/runs/<run_id>-<run_attempt>` y publica solo
los resultados operativos y la evaluación, nunca `evaluator_truth`.

## Simulación de cohortes

`simulate` carga únicamente `load_dev()` y genera sus propensiones ocultas con
un oráculo ajustado exclusivamente sobre esas filas de desarrollo. No lee la
población completa ni la partición reservada. La verdad sintética de la
simulación se escribe separada:

```powershell
$runId = [guid]::NewGuid().ToString("N")
python -m churnshield simulate --seed 101 `
  --output-dir (Join-Path "artifacts\runs" "$runId\public") `
  --evaluator-output-dir (Join-Path "artifacts\runs" "$runId\evaluator_truth")
```

## Evaluación final reservada S01

S01 permanece bloqueado para el uso rutinario. No se ejecuta con pytest,
`dev-eval`, escenarios ni la interfaz. Antes de cargar una sola fila reservada
se requiere:

- commit fijado y árbol Git limpio;
- hashes coincidentes del dataset, partición, `agente.yaml` y `umbrales.yaml`;
- archivo de autorización fuera del repositorio con dos aprobadores distintos,
  referencia y fecha con zona horaria;
- directorio de salida nuevo e inexistente.

Con el protocolo ya integrado en un commit limpio, se puede generar una
plantilla sin leer las filas reservadas:

```powershell
python -m churnshield prepare-final-authorization
```

Completa esa plantilla fuera del repo, con `authorized: true`, ambos nombres,
la referencia y fecha UTC. Ejecuta S01 únicamente cuando la autorización esté
formalmente aprobada:

```powershell
python -m churnshield evaluate-final `
  --confirm-final `
  --authorization-file "$env:TEMP\churnshield-s01-authorization.json" `
  --output-dir "artifacts\final_evaluation"
```

El loader valida y registra el protocolo antes de cargar la partición reservada
y bloquea la reutilización del directorio de salida. Las verificaciones locales
no autorizan ni ejecutan S01.

## Artefactos existentes

Los archivos que ya están en `artifacts/` pueden ser snapshots históricos
versionados. Este código y su documentación no los regeneran ni los eliminan
automáticamente; los comandos de arriba usan destinos explícitos para los
nuevos resultados.

Para iniciar la interfaz local:

```powershell
python -m ui.app
```
