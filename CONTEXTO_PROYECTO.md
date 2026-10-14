# Contexto acumulado — Parcial 2 de Inteligencia Artificial

> Documento de continuidad para que el trabajo pueda retomarse con Claude sin perder decisiones, evidencias ni pendientes.

## 1. Encargo y forma de trabajo

- Proyecto: **Parcial n.º 2 de Inteligencia Artificial**, realizado en pareja.
- Repositorio seleccionado: `Sinoombre/IA-Parcial-2`.
- Rol de este asistente: encargarse de la codificación, la investigación técnica de datasets/entornos y la preparación de pruebas, documentación y evidencia reproducible.
- El usuario indicó que el trabajo se continuará entre dos IAs —este asistente y Claude—. Cada entrega o respuesta debe conservar un contexto acumulado y explicitar qué se hizo, qué falta y cómo continuar.

## 2. Estado verificado el 2026-10-14

- Repositorio clonado en `/home/ubuntu/IA-Parcial-2`.
- Rama activa: `main`, alineada con `origin/main`.
- Commit inicial: `528406b` (`Initial commit`).
- Único archivo existente antes de este contexto: `README.md`, con una descripción general del curso.
- La rúbrica fue recibida como `/home/ubuntu/upload/pasted_content.txt`.
- Todavía **no** están en el repositorio:
  - el documento de la Tarea 2 / diseño inicial;
  - la retroalimentación del docente;
  - la descripción de los dos agentes candidatos;
  - datasets, simuladores, código, dependencias, pruebas o despliegue.

## 3. Requisitos de la rúbrica que deben guiar el desarrollo

### Fase 1 — Selección justificada

1. Comparar los dos agentes con una matriz de decisión ponderada.
2. Usar como mínimo estos seis criterios y pesos sugeridos:
   - Viabilidad técnica: 25 %.
   - Disponibilidad de datos o entorno: 20 %.
   - Medibilidad del desempeño: 20 %.
   - Complejidad de IA: 15 %.
   - Costo y recursos: 10 %.
   - Valor del problema: 10 %.
3. Puntuar cada agente de 1 a 5 y añadir evidencia concreta para cada puntaje.
4. Calcular `S(agente) = Σ(w_i × p_i)`, con pesos expresados como fracción y suma de pesos igual a 1; máximo 5.
5. Hacer análisis de sensibilidad modificando en ±10 puntos porcentuales el criterio más influyente.
6. Justificar el agente elegido y explicar qué se pierde al descartar el otro, en máximo una página.

### Fase 2 — Desarrollo

- El agente implementado debe corresponder al diseño: tipo de agente, ambiente, sensores y actuadores; cualquier cambio se documenta y justifica.
- El código debe hacer identificable el ciclo **percepción → decisión → acción**.
- Debe respetarse la coherencia con el tipo de agente:
  - basado en modelo: estado interno;
  - basado en metas: búsqueda/planificación;
  - basado en utilidad: comparación mediante función de utilidad;
  - de aprendizaje: mejora con experiencia.
- Debe existir un entorno real o simulador mínimo coherente con las percepciones y propiedades del ambiente declarado.
- Cada ejecución debe guardar percepciones, decisiones y acciones con marca de tiempo.
- El repositorio debe ser reproducible: README, dependencias e instrucciones para otro equipo.
- Debe explicarse qué razonamiento implementa el código y qué hace cada herramienta externa. Si se usa LLM: modelo, prompt de sistema y decisiones que dependan de él.

### Fase 3 — Pruebas

- Cada medida SMART del diseño se convierte en al menos una prueba numérica.
- Para cada medida se debe fijar antes de ejecutar:
  - fórmula/regla;
  - umbral de aceptación;
  - escenarios;
  - repeticiones, si hay aleatoriedad.
- Mínimo 10 casos, incluyendo normales, límites y al menos 2 fallas esperadas.
- Si hay aleatoriedad: mínimo 5 ejecuciones por escenario, reportando media y desviación estándar.
- Tabla obligatoria: medida, fórmula, umbral, casos, resultado, cumplimiento y evidencia trazable a logs/capturas/video.
- Incluir análisis de medidas cumplidas/no cumplidas, causas probables y mejoras.

### Entregables

1. PDF del documento de diseño ajustado:
   - correcciones de retroalimentación;
   - tabla inicial de cambios;
   - matriz de decisión y justificación;
   - sección 9 “Desarrollo del agente” antes de conclusiones, que pasan a ser sección 10.
2. Repositorio reproducible con README e instrucciones.
3. Plataforma pública o video de máximo 20 minutos; debe mostrar ejecución desde cero, un escenario típico, una falla esperada y la ejecución de pruebas. En video deben participar ambos integrantes.
4. Logs de pruebas en CSV, JSON o TXT, citados en la tabla.
5. Bitácora de uso de IA.

### Contenido específico de la sección 9

- Arquitectura con diagrama de componentes y correspondencia con los siete componentes IBM de la sección 8; indicar los no implementados y por qué.
- Tabla de tecnologías, versión, uso y justificación frente a alternativas.
- Inputs: nombre, tipo, rango/formato, origen y ejemplo.
- Salidas/acciones y efecto en el entorno.
- Diagrama UML de secuencia con escenario típico y de excepción, participantes completos, mensajes temporales y fragmento `alt` o `loop`.
- Diferencias entre diseño e implementación.
- Pruebas y análisis de resultados.
- Limitaciones y trabajo futuro.
- Enlaces al repositorio y a la evidencia.

## 4. Criterios de calidad para la implementación

- No inventar agentes, datos, resultados ni métricas antes de recibir el diseño base.
- Priorizar un entorno simulado reproducible, determinista mediante semilla cuando sea útil y con una variante estocástica documentada si el diseño lo exige.
- Separar módulos de percepción, decisión, acción, entorno, logging y evaluación para que la trazabilidad sea visible.
- Automatizar las pruebas y generar resultados derivados de los logs, no escribir resultados a mano.
- Mantener una bitácora clara de decisiones, cambios y uso de herramientas externas.

## 5. Pendientes bloqueantes

Antes de seleccionar dataset/entorno o escribir el agente, se necesita recibir al menos:

1. Documento de la Tarea 2 o su contenido completo.
2. Identificación de los dos agentes candidatos.
3. Medidas SMART originales y sus umbrales.
4. Retroalimentación del docente, si existe.
5. Restricciones de lenguaje/plataforma y fecha de entrega, si fueron definidas.

Con esa información se debe continuar en este orden:

1. Extraer y normalizar el diseño de ambos agentes.
2. Investigar y comparar datasets/simuladores viables.
3. Construir la matriz ponderada y el análisis de sensibilidad.
4. Elegir el agente y registrar la decisión.
5. Implementar el agente con logs y pruebas automatizadas.
6. Generar diagramas, documentación, evidencia y PDF.
7. Validar reproducibilidad desde un entorno limpio.

## 6. Último estado comunicado al usuario

Se informó que la rúbrica exige una elección reproducible entre dos agentes, una implementación coherente con el diseño, logs trazables, al menos 10 escenarios con repeticiones cuando aplique, y documento/PDF con evidencia. También se informó que el repositorio actual solo tiene README y que aún faltan la Tarea 2, los dos agentes y la retroalimentación.

## 7. Investigación verificada completada el 2026-10-14

Se verificó el enlace de Kaggle `https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset`. La descarga `data/kaggle/Telco_customer_churn.xlsx` contiene 7043 filas y 33 columnas, con 1869 churners y tasa de fuga 0.2653698708 (26.54%). Las 33 columnas están descritas en `research/informe_datasets_y_entornos.txt`. Kaggle muestra licencia "Other (specified in description)"; no debe afirmarse que el dataset tiene licencia Apache-2.0, aunque el repositorio envolvente de IBM sí declara Apache-2.0. La copia CSV de IBM en `data/Telco-Customer-Churn.csv` tiene 7043 filas y 21 columnas, por lo que no es la misma variante enriquecida de 33 columnas.

La exploración detectó 11 faltantes en `Total Charges` y 5174 faltantes esperables en `Churn Reason`. Para evitar fuga se excluyeron `Churn Label`, `Churn Value`, `Churn Score`, `Churn Reason`, identificadores, campos de reporte y ubicación detallada. `CLTV` se consideró disponible antes de la decisión y se incluyó como covariable legítima/valor relativo, pero la fuente no publica su unidad monetaria ni fórmula.

Se compararon regresión logística y Random Forest con partición externa estratificada de prueba del 20% (semilla 2026), división interna estratificada y cinco semillas (11, 22, 33, 44, 55). En test, la regresión logística obtuvo AUC media 0.8550 (sd 0.0013), Precision@15% media 0.7255 (sd 0.0062) y Recall@15% media 0.4112 (sd 0.0035). Random Forest obtuvo AUC 0.8538 (sd 0.0028), Precision@15% 0.7000 (sd 0.0144) y Recall@15% 0.3968 (sd 0.0082). La tasa base fue 0.2654. La meta original Precision@15% >= 0.75 no se alcanzó por estas líneas base en el test intacto; no se debe bajar el umbral después de observar el test.

Alternativas verificadas: el benchmark de Orange Belgium para churn uplift tiene 11896 muestras, 178 características anonimizadas, tres campañas entre septiembre y diciembre de 2020, control/tratamiento y ventana de resultado de dos meses; licencia del artículo arXiv CC BY-NC-ND 4.0. KDD Cup 2009 tiene 50000 ejemplos de entrenamiento y 50000 de prueba, objetivos separados de churn, appetency y upselling, pero no documenta historial mensual identificado por cliente. Cell2Cell solo apareció en fuentes secundarias y no se confirmó su estructura longitudinal/licencia primaria.

No se confirmó un dataset público grande y listo que combine etiquetas de phishing con encabezados y resultados SPF, DKIM y DMARC observados para cada mensaje. SpamAssassin conserva todos los headers y 6047 mensajes, pero es antiguo, principalmente spam/ham y no confirma Authentication-Results sistemáticos. TREC tampoco confirma phishing + SPF/DKIM/DMARC. La colección Figshare de 11 datasets tiene CC BY 4.0, pero la página no demuestra autenticación técnica completa por mensaje. Esto no justifica modificar los puntajes de PhishGuard.

Informe completo: `research/informe_datasets_y_entornos.txt`. Resultados reproducibles: `research/exploratory_metrics_by_seed.csv`, `research/exploratory_metrics_summary.csv`, `research/exploratory_metadata.json`. No se implementó todavía el agente, conforme a la instrucción explícita.

## 8. Próximo paso recomendado

Claude puede redactar la justificación de una página, la tabla de cambios y la reformulación formal de Precision@15%. Después corresponde implementar ChurnShield con el dataset sin columnas de fuga, un simulador mensual explícitamente sintético, función de utilidad con parámetros marcados como supuestos, logs y pruebas automatizadas. El simulador debe conservar la semilla, incluir aleatoriedad y separar claramente percepción, decisión, acción y actualización del entorno.
