# Selection Module

## Índice Del Documento

1. Inicio Recomendado
2. Mapa De La Serie Piloto
3. Descripción Del Módulo
4. Estructura
5. Documentación
6. Sistema De Diagramas
7. Flujo Orgánico
8. Principios De Diseño
9. Convenciones

## Inicio Recomendado

La documentación nueva de `selection` empieza en
[00_index.md](00_index.md). Ese documento define el orden de lectura y explica
qué archivo consultar según la necesidad del lector.

Este `README.md` funciona como puerta de entrada. El índice operativo del módulo
vive en `00_index.md`.

## Mapa De La Serie Piloto

| Orden | Documento | Rol |
|---:|---|---|
| 00 | [00_index.md](00_index.md) | Ruta secuencial de lectura. |
| 01 | [01_conceptual_model.md](01_conceptual_model.md) | Modelo conceptual de selección. |
| 02 | [02_architecture.md](02_architecture.md) | Arquitectura ordenada del módulo. |
| 03 | [03_data_contracts.md](03_data_contracts.md) | Entradas, salidas y estados auditables. |
| 04 | [04_fundamental_selection_article.md](04_fundamental_selection_article.md) | Artículo sobre selección fundamental. |
| 05 | [05_correlation_selection_article.md](05_correlation_selection_article.md) | Artículo sobre selección por correlación. |
| 06 | [06_scoring_model.md](06_scoring_model.md) | Modelo de señales, pesos y score. |
| 07 | [07_workflows.md](07_workflows.md) | Flujos prácticos recomendados. |
| 08 | [08_api_reference.md](08_api_reference.md) | API narrativa. |
| 09 | [09_validation_and_edge_cases.md](09_validation_and_edge_cases.md) | Validación y límites. |
| 10 | [10_glossary.md](10_glossary.md) | Glosario del módulo. |
| 11 | [11_learned_fundamental_scoring.md](11_learned_fundamental_scoring.md) | Scoring aprendido con XGBoost. |

## Descripción Del Módulo

El módulo `src.selection` concentra la lógica de selección de activos antes de
pasar a `optimization` y a las etapas posteriores de construcción,
backtesting, análisis de desempeño y reporte de riesgo. Su objetivo es
responder una pregunta previa a la construcción del portafolio:

> De un universo inicial de compañías, ¿cuáles merecen entrar al conjunto de
> candidatos que después se va a optimizar o analizar?

Actualmente el módulo tiene dos rutas principales de selección y una extensión
temporal de análisis:

- Selección por correlación, útil para construir universos diversificados.
- Selección fundamental, útil para rankear compañías con criterios de `value`
  investing o `growth` investing usando información de Yahoo Finance.
- Análisis fundamental temporal, útil para revisar la evolución trimestral o
  anual de métricas y scores.
- Scoring fundamental granular: cada tesis puede ponderar el nivel reciente de
  una métrica, su cambio contra el periodo anterior, su cambio interanual y su
  cambio histórico promedio.
- Catálogo fundamental: `FUNDAMENTAL_METRIC_SIGNAL_SPECS` expone las métricas
  disponibles como componentes `MetricSignalSpec`.
- Scoring fundamental aprendido: construye targets forward, entrena modelos
  tipo XGBoost por grupo y convierte impactos en pesos compatibles con el
  scorer existente.

## Estructura

```text
src/selection/
├── __init__.py
├── correlation_selector.py
├── fundamentals.py
├── fundamental_panel.py
├── fundamental_metrics.py
├── fundamental_scorers.py
├── fundamental_selector.py
├── fundamental_targets.py
├── learned_fundamental_scorers.py
├── learned_fundamental_selector.py
└── xgboost_fundamental_model.py
```

## Documentación

- [00_index.md](00_index.md): punto inicial recomendado y mapa secuencial de la
  documentación nueva.
- [01_conceptual_model.md](01_conceptual_model.md): explicación conceptual del
  módulo y de sus límites.
- [04_fundamental_selection_article.md](04_fundamental_selection_article.md):
  artículo explicativo de selección fundamental, value, growth y scoring.
- [05_correlation_selection_article.md](05_correlation_selection_article.md):
  artículo explicativo de selección por correlación y diversificación.
- [06_scoring_model.md](06_scoring_model.md): detalle del sistema
  `MetricSignalSpec`, normalización, pesos y `score_coverage`.
- [11_learned_fundamental_scoring.md](11_learned_fundamental_scoring.md):
  flujo para aprender ponderadores fundamentales con XGBoost.
- [architecture.md](architecture.md): arquitectura lineal del módulo, rutas,
  contratos de salida y relación con los diagramas.
- [api_reference.md](api_reference.md): referencia de clases, atributos, métodos y funciones.
- [workflows.md](workflows.md): casos de uso y flujos de trabajo recomendados.
- [diagrams/README.md](diagrams/README.md): diagramas `.drawio` del módulo y
  links directos para abrirlos en diagrams.net.

## Sistema De Diagramas

La documentación visual vive dentro de [diagrams](diagrams/README.md) y sigue
la misma lógica metodológica usada para el flujo completo de portfolio
management:

```text
selection -> optimization -> portfolio construction -> execution/backtesting -> performance analysis -> risk/reporting
```

Los entregables visuales son:

1. `selection_module_architecture.drawio`: vista universo del módulo.
2. `selection_class_architecture.drawio`: vista de clases, funciones y
   relaciones.
3. `selection_workflow.drawio`: vista orgánica de uso.
4. `open_in_diagrams_net.md`: acceso directo a copias editables en diagrams.net.

Estos diagramas son la referencia visual para entender la bifurcación entre
ruta fundamental y ruta de correlación, así como su convergencia en una lista
corta de tickers para `optimization`.

## Flujo Orgánico

El uso más natural del módulo empieza con un universo amplio de tickers y termina
con una lista compacta de candidatos.

```python
from src.selection import FundamentalSelector

selector = FundamentalSelector(strategy="value")
ranking = selector.rank(["AAPL", "MSFT", "NVDA", "KO"])
selected = selector.select_top(ranking, top_k=3)
```

El resultado `selected` es un `DataFrame` con las compañías mejor rankeadas, sus
métricas fundamentales y su `fundamental_score`. Ese resultado puede alimentar
los módulos posteriores del proyecto:

```python
selected_tickers = selected["ticker"].tolist()
```

A partir de ahí, `selected_tickers` puede alimentar `optimization`; después el
flujo continúa hacia portfolio construction, execution/backtesting, performance
analysis y risk/reporting.

Para revisar el score a través del tiempo:

```python
score_history = selector.rank_over_time(
    ["AAPL", "MSFT", "NVDA", "KO"],
    frequency="quarterly",
    trailing_periods=4,
)
```

Para explicar la selección final:

```python
report = selector.selection_report(ranking, top_k=3)
```

`report` resume los tickers seleccionados, las métricas utilizadas, sus pesos y
los componentes de score. En los scores granulares también muestra
`score_coverage`, útil para ver cuánta información disponible respaldó el score
de cada compañía.

## Principios De Diseño

El diseño del módulo sigue cuatro principios:

1. Separar adquisición de datos, cálculo de métricas y ranking.
2. Separar la tesis fundamental de la diversificación por correlación.
3. Mantener funciones puras para que las métricas y scores sean testeables.
4. Exponer una API simple para notebooks e investigación exploratoria.

La separación es intencional. Yahoo Finance puede cambiar, devolver campos
faltantes o comportarse de forma irregular. Por eso `fundamentals.py` aísla la
descarga, mientras que `fundamental_metrics.py` y `fundamental_scorers.py`
trabajan sobre estructuras de `pandas`.

El proveedor de Yahoo Finance mantiene una caché por ticker, rango de fechas y
campo de precio. Esto evita repetir consultas cuando se calculan métricas,
historiales y reportes sobre el mismo universo.

## Convenciones

La documentación usa una estructura similar a NumPy y pandas:

- Descripción breve de cada clase o función.
- Parámetros.
- Atributos.
- Métodos.
- Retornos.
- Notas.
- Ejemplos.

Los nombres públicos usan `snake_case` para funciones y métodos, y `CamelCase`
para clases, siguiendo PEP 8.
