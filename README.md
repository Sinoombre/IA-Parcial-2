# IA-Parcial-2

<div align="center">

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Estado](https://img.shields.io/badge/estado-en_desarrollo-yellow.svg)
![Ruff](https://img.shields.io/badge/linter-ruff-ff6f00.svg)
![Tests](https://img.shields.io/badge/tests-pytest-0A9ED9.svg)

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-e52626.svg?logo=xgboost)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-2470b8.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.3-000.svg?logo=flask)

</div>

Proyecto de predicción y simulación de retención para el segundo parcial de Inteligencia Artificial.

El proyecto implementa un sistema llamado ChurnShield que predice la probabilidad de que un cliente cancele su servicio (churn), selecciona a los clientes a los que se les debe hacer una campaña de retención según una función de utilidad, simula cohortes de clientes y expone una interfaz web de predicción. Incluye además un protocolo de evaluación final reservada (S01) que permanece bloqueado hasta que se cumple un proceso de autorización formal.

## Tabla de contenidos

1. [Qué hace el proyecto](#1-qué-hace-el-proyecto)
2. [Arquitectura y componentes principales](#2-arquitectura-y-componentes-principales)
3. [Requisitos previos](#3-requisitos-previos)
4. [Preparar el entorno paso a paso](#4-preparar-el-entorno-paso-a-paso)
5. [Configuración del modelo (config/agent.yaml)](#5-configuración-del-modelo-configagentyaml)
6. [Entrenar y regenerar el modelo](#6-entrenar-y-regenerar-el-modelo)
7. [Ejecución paso a paso del proyecto](#7-ejecución-paso-a-paso-del-proyecto)
   - [7.1 Regenerar resultados públicos](#71-regenerar-resultados-públicos)
   - [7.2 Experimentos de comparación](#72-experimentos-de-comparación)
   - [7.3 Simulación de cohortes](#73-simulación-de-cohortes)
   - [7.4 Iniciar la interfaz local](#74-iniciar-la-interfaz-local)
8. [Evaluación final reservada S01 y protocolo de autorización](#8-evaluación-final-reservada-s01-y-protocolo-de-autorización)
9. [Pruebas y calidad de código](#9-pruebas-y-calidad-de-código)
10. [Organización de artifacts (público vs privado)](#10-organización-de-artifacts-público-vs-privado)
11. [Estructura del repositorio](#11-estructura-del-repositorio)
12. [Notas importantes y restricciones](#12-notas-importantes-y-restricciones)
13. [Contacto y créditos](#13-contacto-y-créditos)

## 1. Qué hace el proyecto

ChurnShield es un proyecto de Inteligencia Artificial centrado en la retención de clientes. Su objetivo principal es predecir qué clientes tienen mayor probabilidad de cancelar su servicio (churn) y decidir a cuáles de ellos dirigir una campaña de retención, maximizando el valor esperado de la acción.

El sistema funciona con un conjunto de exactamente nueve predictores permitidos definidos en `config/agent.yaml`. Las ocho columnas bajo `excluded_by_design` no se usan como características del modelo, y se excluyen además campos sensibles, de fuga, identificadores y geográficos para evitar sesgos y fuga de información.

El flujo general del proyecto es:

1. Entrenamiento y regeneración del modelo de desarrollo (`dev-eval`).
2. Simulación de cohortes con un oráculo ajustado solo sobre filas de desarrollo (`simulate`).
3. Ejecución de escenarios sintéticos S02 a S10 con cinco semillas cada uno (`scenarios --all --all-seeds`).
4. Evaluación de escenarios contra el conjunto esperado (`evaluate`).
5. Ejecución de experimentos comparativos de modelos (`experiments/*.py`).
6. Consumo del modelo entrenado mediante una interfaz web local (`ui.app`).
7. Opcionalmente, ejecución de la evaluación final reservada S01 bajo protocolo de autorización (`evaluate-final`).

El modelo de desarrollo es un stacking regularizado que combina regresión logística, LightGBM, XGBoost y CatBoost. Se conserva una regresión logística independiente como referencia de comparación. El model card y el manifiesto registran el conjunto de features y los hashes del dataset, la partición, la configuración y los umbrales.

## 2. Arquitectura y componentes principales

El paquete `src/churnshield` está organizado en módulos con responsabilidades separadas para que la trazabilidad sea visible:

- `data.py`: carga y preparación del dataset. Expone `load_dev()` para usar solo las filas de desarrollo.
- `features.py`: construcción de características a partir de los nueve predictores permitidos.
- `model.py`: definición y entrenamiento del modelo de stacking.
- `dev_eval.py`: comando para entrenar y regenerar el modelo de desarrollo.
- `perception.py`: conversión de datos de entrada en percepciones del agente.
- `decision.py`: lógica de decisión basada en utilidad para seleccionar clientes.
- `oracle.py`: oráculo ajustado exclusivamente sobre filas de desarrollo para generar propensiones ocultas en simulación.
- `simulation.py`: simulación de cohortes con ciclos, semillas y actualización de creencias.
- `crm.py`: gestión de la campaña de retención y sus resultados (aceptado, rechazado, error, timeout).
- `robustness.py`: manejo de fallas esperadas y registro de casos de falla.
- `evaluation.py`: evaluación de escenarios y generación de resúmenes.
- `final_eval.py`: evaluación final reservada S01 con validación de protocolo.
- `protocol.py`: validaciones de protocolo (árbol Git limpio, hashes, autorización, directorio de salida).
- `settings.py`: configuración global y rutas de artifacts.
- `metrics.py`: cálculo de métricas (Precisión@15, Recall@15, AUC, Brier, lift).
- `scenario_report.py`: generación de reportes por escenario.
- `stacking.py`: comparación de variantes de stacking.
- `__main__.py`: punto de entrada de la consola (`python -m churnshield`).

La carpeta `ui` contiene una aplicación Flask (`app.py`) con templates y static que sirve como interfaz local de predicción. Antes de predecir, verifica el model card; si el modelo falta, la tarjeta es inválida o las features no coinciden, responde con HTTP 503 en vez de inventar valores.

La carpeta `experiments` contiene scripts de comparación: `logistic_regression.py`, `random_forest.py` y `stacking.py`.

## 3. Requisitos previos

- Python 3.12 (el proyecto se prueba con esta versión).
- Acceso a PowerShell (para los comandos de ejecución mostrados en este documento).
- Conexión a Internet para instalar las dependencias la primera vez.
- Git instalado si se desea ejecutar el protocolo S01 (se requiere árbol Git limpio y commit fijado).

No se requieren credenciales de cloud ni claves de API externas. Todo se ejecuta localmente.

## 4. Preparar el entorno paso a paso

Desde PowerShell, en la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH = "src"
```

Esto crea un entorno virtual en `.venv`, lo activa, actualiza pip, instala las dependencias de producción y de desarrollo, y establece `PYTHONPATH` apuntando a `src` para que el paquete `churnshield` sea importable.

Si PowerShell bloquea la activación del entorno virtual, ejecute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Después de esto, puede verificar que el entorno está listo con:

```powershell
python --version
python -m churnshield --help
```

Las dependencias principales instaladas son:

- numpy, pandas, scipy, scikit-learn, statsmodels para manipulación de datos y modelado.
- lightgbm, xgboost, catboost para los modeladores del stacking.
- pyyaml para lectura de configuración.
- joblib para serialización del modelo.
- Flask para la interfaz web.
- matplotlib, seaborn, openpyxl, jupyter para análisis y experimentos.
- pytest y ruff para pruebas y calidad de código.

## 5. Configuración del modelo (config/agent.yaml)

`config/agent.yaml` es la fuente canónica de predictores y parámetros del agente. El modelo, la evaluación y la interfaz usan exactamente sus nueve campos `allowed_predictors`:

1. Partner
2. Dependents
3. Internet Service
4. Contract
5. Paperless Billing
6. Payment Method
7. Tenure Months
8. Monthly Charges
9. Total Charges

Las ocho columnas bajo `excluded_by_design` no se usan como features:

1. Phone Service
2. Multiple Lines
3. Online Security
4. Online Backup
5. Device Protection
6. Tech Support
7. Streaming TV
8. Streaming Movies

Además se excluyen campos sensibles (Gender, Senior Citizen), campos de fuga (Churn Label, Churn Value, Churn Score, Churn Reason), identificadores y geográficos (CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude). CLTV se usa solo como utilidad, no como predictor.

Otros parámetros clave en `config/agent.yaml`:

- `data.target`: "Churn Value".
- `selection.capacity_fraction`: 0.15 (se selecciona el 15 % superior de clientes por utilidad).
- `utility.formula`: p_churn * r * v_cliente - cost.
- `model.base`: stacked_logistic_lightgbm_xgboost_catboost.
- `model.cv_folds`: 5.
- `model.selection_metric`: precision_at_15.
- `simulation.master_seed`: 1702.
- `simulation.repetition_seeds`: [101, 102, 103, 104, 105].

El model card y el manifiesto registran el conjunto de features y los hashes del dataset, la partición, la configuración y los umbrales. `config/thresholds.yaml` registra las medidas formales M1 (Precisión@15 con umbral 0.75), M2 (trazabilidad al 100 %) y M3 (manejo de fallas al 100 %).

## 6. Entrenar y regenerar el modelo

Para entrenar el modelo de desarrollo y regenerar los artifacts que consume la interfaz, ejecute:

```powershell
python -m churnshield dev-eval
```

Esto entrena la variante de stacking con CatBoost y regenera el modelo en `artifacts/public/dev_eval/stacking/`. Antes de predecir, la interfaz verifica el model card; si el modelo falta, la tarjeta es inválida o las features no coinciden, responde con HTTP 503 en vez de inventar valores.

La regresión logística independiente se conserva como referencia de comparación y no se sobrescribe con este comando.

## 7. Ejecución paso a paso del proyecto

Todos los comandos a continuación se ejecutan desde PowerShell, en la raíz del repo, con el entorno ya preparado (véase sección 4).

### 7.1 Regenerar resultados públicos

Los escenarios S02 a S10 usan fixtures sintéticos. `--all-seeds` ejecuta cinco semillas por escenario. La salida pública y la verdad privada tienen rutas separadas por defecto.

```powershell
python -m churnshield dev-eval
python -m churnshield simulate --seed 101
python -m churnshield scenarios --all --all-seeds
python -m churnshield evaluate `
  --expected-scenarios S02 S03 S04 S05 S06 S07 S08 S09 S10 `
  --expected-seeds 101 102 103 104 105
```

`evaluate` lee `artifacts/public/scenarios/` y guarda el resumen en `artifacts/public/scenario_evaluation/evaluation.json`. Rechaza carpetas sin ejecuciones y valida el conjunto exacto escenario/semilla indicado. El workflow manual usa un directorio temporal por ejecución, fuera de `artifacts/`, y publica solo resultados operativos; nunca sube la verdad del evaluador.

### 7.2 Experimentos de comparación

```powershell
python experiments\logistic_regression.py
python experiments\random_forest.py
python experiments\stacking.py
```

`stacking.py` compara una regresión logística regularizada con dos variantes de stacking (LightGBM + XGBoost, con y sin CatBoost), usando solo desarrollo, las nueve features permitidas, tres semillas y precisión en el 15 % superior. Sus resultados se escriben en `artifacts/public/experiments/stacking/stacking_comparison.json`. `dev-eval` entrena la variante con CatBoost y regenera el modelo que consume la interfaz; el experimento no ejecuta S01.

### 7.3 Simulación de cohortes

`simulate` carga únicamente `load_dev()` y genera sus propensiones ocultas con un oráculo ajustado exclusivamente sobre esas filas de desarrollo. No lee la población completa ni la partición reservada.

```powershell
python -m churnshield simulate --seed 101
```

Los resultados públicos quedan en `artifacts/public/cohort_simulation/`. La verdad sintética del evaluador se guarda en `artifacts/private/evaluator_truth/cohort_simulation/`.

### 7.4 Iniciar la interfaz local

Para iniciar la interfaz web de predicción:

```powershell
python -m ui.app
```

La interfaz usa el modelo de desarrollo de `artifacts/public/dev_eval/stacking/`. Antes de predecir verifica el model card; si el modelo falta, la tarjeta es inválida o las features no coinciden, responde con HTTP 503 en vez de inventar valores.

## 8. Evaluación final reservada S01 y protocolo de autorización

S01 permanece bloqueado para el uso rutinario. No se ejecuta con pytest, `dev-eval`, escenarios ni la interfaz. Antes de cargar una sola fila reservada se requiere:

- commit fijado y árbol Git limpio;
- hashes coincidentes del dataset, partición, `agent.yaml` y `thresholds.yaml`;
- archivo de autorización fuera del repositorio con dos aprobadores distintos, referencia y fecha con zona horaria;
- directorio de salida nuevo e inexistente.

Con el protocolo ya integrado en un commit limpio, se puede generar una plantilla sin leer las filas reservadas:

```powershell
python -m churnshield prepare-final-authorization
```

Complete esa plantilla fuera del repo, con `authorized: true`, ambos nombres, la referencia y fecha UTC. Ejecute S01 únicamente cuando la autorización esté formalmente aprobada:

```powershell
python -m churnshield evaluate-final `
  --confirm-final `
  --authorization-file "$env:TEMP\churnshield-s01-authorization.json" `
  --output-dir "artifacts\private\final_evaluation"
```

El loader valida y registra el protocolo antes de cargar la partición reservada y bloquea la reutilización del directorio de salida. Las verificaciones locales no autorizan ni ejecutan S01.

## 9. Pruebas y calidad de código

Ejecutar toda la suite de pruebas:

```powershell
python -m pytest -q
```

Verificar calidad con Ruff:

```powershell
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

El workflow **Code quality** ejecuta Ruff y toda la suite de pytest en cada integración.

## 10. Organización de artifacts (público vs privado)

`artifacts/public/` contiene resultados operativos reproducibles que se pueden revisar y versionar: métricas de desarrollo, modelo de desarrollo, salidas sintéticas y resultados de experimentos.

`artifacts/private/` contiene verdad del evaluador, registros de acceso y resultados reservados. Git ignora toda esa carpeta para no publicar etiquetas ni datos de evaluación.

El índice de partición en `artifacts/public/data/` es metadato de protocolo necesario para validar las cargas; el dataset fuente está en `data/`. La separación evita mezclar salidas antiguas de distintas ejecuciones y evita que la verdad del evaluador se publique por accidente.

Rutas principales de resultados:

- `artifacts/public/dev_eval/stacking/`: modelo de desarrollo y model card.
- `artifacts/public/scenarios/`: salidas de escenarios S02 a S10.
- `artifacts/public/scenario_evaluation/evaluation.json`: resumen de evaluación de escenarios.
- `artifacts/public/cohort_simulation/`: resultados públicos de simulación de cohortes.
- `artifacts/public/experiments/stacking/stacking_comparison.json`: resultados del experimento stacking.
- `artifacts/private/evaluator_truth/cohort_simulation/`: verdad sintética del evaluador (privada).
- `artifacts/private/final_evaluation/`: resultados de S01 (privada, solo si se autoriza).

Se retiraron las carpetas de resultados duplicadas o históricas y se volvieron a generar las salidas operativas desde el código actual. No se borró el dataset fuente ni el índice de partición; este último se reubicó bajo `artifacts/public/data/`.

## 11. Estructura del repositorio

```
IA-Parcial-2/
+-- README.md
+-- PROJECT_CONTEXT.md
+-- requirements.txt
+-- requirements-dev.txt
+-- ruff.toml
+-- .gitignore
+-- config/
|   +-- agent.yaml
|   +-- seed.yaml
|   +-- thresholds.yaml
+-- data/
|   +-- Telco_customer_churn.xlsx
|   +-- EDA.ipynb
+-- src/
|   +-- churnshield/
|   |   +-- __main__.py
|   |   +-- data.py
|   |   +-- features.py
|   |   +-- model.py
|   |   +-- dev_eval.py
|   |   +-- perception.py
|   |   +-- decision.py
|   |   +-- oracle.py
|   |   +-- simulation.py
|   |   +-- crm.py
|   |   +-- robustness.py
|   |   +-- evaluation.py
|   |   +-- final_eval.py
|   |   +-- protocol.py
|   |   +-- settings.py
|   |   +-- metrics.py
|   |   +-- scenario_report.py
|   |   +-- stacking.py
|   |   +-- scenarios/
|   |   +-- simulation/
+-- experiments/
|   +-- logistic_regression.py
|   +-- random_forest.py
|   +-- stacking.py
+-- ui/
|   +-- app.py
|   +-- static/
|   +-- templates/
+-- tests/
|   +-- agent/
|   +-- evaluation/
|   +-- experiments/
|   +-- final/
|   +-- modeling/
|   +-- scenarios/
|   +-- simulation/
|   +-- ui/
+-- artifacts/
|   +-- public/
|   +-- scenario_evaluation/
```

## 12. Notas importantes y restricciones

- Solo se permiten los nueve predictores en `config/agent.yaml`. Las ocho columnas de `excluded_by_design` no se usan como features.
- La interfaz responde HTTP 503 si el modelo falta, el model card es inválido o las features no coinciden. Nunca inventa valores.
- `dev-eval` entrena la variante con CatBoost y regenera el modelo de desarrollo. No ejecuta S01.
- `evaluate` rechaza carpetas sin ejecuciones y valida el conjunto exacto escenario/semilla indicado.
- El workflow manual usa un directorio temporal por ejecución, fuera de `artifacts/`, y publica solo resultados operativos. Nunca sube la verdad del evaluador.
- S01 permanece bloqueado para uso rutinario. Requiere árbol Git limpio, hashes coincidentes, archivo de autorización con dos aprobadores y directorio de salida nuevo.
- Las verificaciones locales no autorizan ni ejecutan S01.
- El proyecto se prueba con Python 3.12. Otras versiones no están garantizadas.

## 13. Contacto y créditos

Proyecto realizado para el segundo parcial de Inteligencia Artificial. Para dudas sobre la ejecución, consulte `PROJECT_CONTEXT.md` o los comentarios en el código fuente de cada módulo.
