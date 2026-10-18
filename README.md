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

`config/agent.yaml` es la fuente canónica de predictores. El modelo, la
evaluación y la interfaz usan exactamente sus nueve campos `allowed_predictors`;
las ocho columnas bajo `excluded_by_design` no se usan como features. El modelo
de desarrollo es un stacking regularizado de regresión logística, LightGBM,
XGBoost y CatBoost. La regresión logística independiente se conserva como
referencia de comparación. El model card y el manifiesto registran el conjunto
de features y los hashes del dataset, la partición, la configuración y los
umbrales.

La interfaz usa el modelo de desarrollo de
`artifacts/public/dev_eval/stacking/`. Antes de
predecir verifica el model card; si el modelo falta, la tarjeta es inválida o
las features no coinciden, responde con HTTP 503 en vez de inventar valores.
Para entrenar y regenerar el modelo, ejecuta `dev-eval`:

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

## Organización de artifacts

`artifacts/public/` contiene resultados operativos reproducibles que se pueden
revisar y versionar: métricas de desarrollo, modelo de desarrollo, salidas
sintéticas y resultados de experimentos. `artifacts/private/` contiene verdad
del evaluador, registros de acceso y resultados reservados; Git ignora toda
esa carpeta para no publicar etiquetas ni datos de evaluación.

El índice de partición en `artifacts/public/data/` es metadato de protocolo
necesario para validar las cargas; el dataset fuente está en `data/`. La
separación evita mezclar salidas antiguas de distintas ejecuciones y evita que
la verdad del evaluador se publique por accidente. Se retiraron las carpetas
de resultados duplicadas/históricas y se volvieron a generar las salidas
operativas desde el código actual. No se borró el dataset fuente ni el índice
de partición; este último se reubicó bajo `artifacts/public/data/`.

## Regenerar resultados públicos

Los escenarios S02–S10 usan fixtures sintéticos; `--all-seeds` ejecuta cinco
semillas por escenario. La salida pública y la verdad privada tienen rutas
separadas por defecto. Desde PowerShell, con el entorno preparado:

```powershell
python -m churnshield dev-eval
python -m churnshield simulate --seed 101
python -m churnshield scenarios --all --all-seeds
python -m churnshield evaluate `
  --expected-scenarios S02 S03 S04 S05 S06 S07 S08 S09 S10 `
  --expected-seeds 101 102 103 104 105
python experiments\logistic_regression.py
python experiments\random_forest.py
python experiments\stacking.py
```

`stacking.py` compara una regresión logística regularizada con dos variantes de
stacking (LightGBM + XGBoost, con y sin CatBoost), usando únicamente desarrollo,
las nueve features permitidas, tres semillas y precisión en el 15% superior.
Sus resultados se escriben en
`artifacts/public/experiments/stacking/stacking_comparison.json`. `dev-eval`
entrena la variante con CatBoost y regenera el modelo que consume la interfaz;
el experimento no ejecuta S01.

`evaluate` lee `artifacts/public/scenarios/` y guarda el resumen en
`artifacts/public/scenario_evaluation/evaluation.json`. Rechaza carpetas sin
ejecuciones y valida el conjunto exacto escenario/semilla indicado. El workflow
manual usa un directorio temporal por ejecución, fuera de `artifacts/`, y
publica solo resultados operativos; nunca sube la verdad del evaluador.

## Simulación de cohortes

`simulate` carga únicamente `load_dev()` y genera sus propensiones ocultas con
un oráculo ajustado exclusivamente sobre esas filas de desarrollo. No lee la
población completa ni la partición reservada. La verdad sintética de la
simulación se escribe en la carpeta privada:

Los resultados públicos quedan en `artifacts/public/cohort_simulation/`; la
verdad sintética del evaluador se guarda en
`artifacts/private/evaluator_truth/cohort_simulation/`.

## Evaluación final reservada S01

S01 permanece bloqueado para el uso rutinario. No se ejecuta con pytest,
`dev-eval`, escenarios ni la interfaz. Antes de cargar una sola fila reservada
se requiere:

- commit fijado y árbol Git limpio;
- hashes coincidentes del dataset, partición, `agent.yaml` y `thresholds.yaml`;
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
  --output-dir "artifacts\private\final_evaluation"
```

El loader valida y registra el protocolo antes de cargar la partición reservada
y bloquea la reutilización del directorio de salida. Las verificaciones locales
no autorizan ni ejecutan S01.

Para iniciar la interfaz local:

```powershell
python -m ui.app
```
