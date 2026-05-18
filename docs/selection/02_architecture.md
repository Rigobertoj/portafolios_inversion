# Selection: Arquitectura Del Módulo

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Vista General
4. Responsabilidades Por Archivo
5. Arquitectura Fundamental
6. Arquitectura De Correlación
7. Diagramas
8. Handoff A Otros Módulos

## Mapa De La Serie

| Orden | Documento | Rol | Cuándo Leerlo |
|---:|---|---|---|
| 00 | `00_index.md` | Entrada secuencial | Siempre primero. |
| 01 | `01_conceptual_model.md` | Modelo conceptual | Antes de usar o modificar `selection`. |
| 02 | `02_architecture.md` | Arquitectura del módulo | Antes de tocar código. |
| 03 | `03_data_contracts.md` | Contratos de datos | Cuando se integran entradas o salidas. |
| 04 | `04_fundamental_selection_article.md` | Artículo fundamental | Para entender value, growth y scoring. |
| 05 | `05_correlation_selection_article.md` | Artículo correlación | Para entender diversificación estadística. |
| 06 | `06_scoring_model.md` | Modelo de score | Para ajustar métricas, señales o pesos. |
| 07 | `07_workflows.md` | Flujos prácticos | Para ejecutar casos de uso. |
| 08 | `08_api_reference.md` | API narrativa | Para consultar clases, métodos y atributos. |
| 09 | `09_validation_and_edge_cases.md` | Validación | Para pruebas, errores y límites. |
| 10 | `10_glossary.md` | Glosario | Para unificar vocabulario. |

## Propósito

Este documento explica cómo está organizado `src.selection` y cómo se relacionan
sus archivos principales. La versión heredada vive en `architecture.md`; esta
versión agrega una ruta secuencial y una lectura orientada a responsabilidades.

## Vista General

```text
src/selection/
├── __init__.py
├── fundamentals.py
├── fundamental_targets.py
├── fundamental_panel.py
├── fundamental_metrics.py
├── fundamental_scorers.py
├── fundamental_selector.py
├── learned_fundamental_scorers.py
├── learned_fundamental_selector.py
├── xgboost_fundamental_model.py
└── correlation_selector.py
```

La arquitectura tiene dos rutas que convergen en una lista de candidatos:

```text
universo de tickers
        |
        v
decisión metodológica
        |
        +-- ruta fundamental -> métricas -> score -> ranking -> selected_tickers
        |
        +-- ruta fundamental aprendida -> panel -> XGBoost -> pesos -> score
        |
        +-- ruta correlación -> retornos -> correlación -> ranking -> selected_tickers
```

## Responsabilidades Por Archivo

| Archivo | Responsabilidad | Tipo De Capa |
|---|---|---|
| `__init__.py` | Define la API pública de `src.selection`. | Interfaz pública |
| `fundamentals.py` | Descarga y normaliza datos fundamentales de Yahoo Finance. | Datos |
| `fundamental_targets.py` | Alinea fundamentales con retornos forward de precio y dividendos. | Targets |
| `fundamental_panel.py` | Construye paneles históricos con señales `metric__signal`. | Research |
| `fundamental_metrics.py` | Convierte estados financieros en métricas comparables. | Transformación |
| `fundamental_scorers.py` | Define señales, pesos, normalizadores y score compuesto. | Modelo |
| `fundamental_selector.py` | Orquesta proveedor, métricas, scoring, ranking y reportes. | Workflow |
| `xgboost_fundamental_model.py` | Ajusta modelos no lineales y reporta impactos por señal. | Research |
| `learned_fundamental_scorers.py` | Convierte impactos aprendidos en `FundamentalScoreConfig`. | Modelo |
| `learned_fundamental_selector.py` | Aplica configs aprendidas por sector o industria. | Workflow |
| `correlation_selector.py` | Rankea activos por correlación y construye candidatos diversificados. | Ruta estadística |

## Arquitectura Fundamental

La ruta fundamental está diseñada como una separación de responsabilidades:

```text
FundamentalSelector
  -> YahooFundamentalsProvider
  -> FundamentalData
  -> build_metrics_frame
  -> score_fundamentals
  -> selection_report
```

## Arquitectura Fundamental Aprendida

La ruta aprendida agrega una etapa de research antes del scorer operativo:

```text
FundamentalLearningPanelBuilder
  -> build_metric_history_frame
  -> build_forward_return_targets
  -> add_signal_features
  -> XGBoostFundamentalModel.fit
  -> impact_report_
  -> LearnedScoreConfigFactory
  -> LearnedFundamentalSelector
  -> score_fundamentals
```

La dependencia de `xgboost` queda aislada en `xgboost_fundamental_model.py`.
Los selectores value/growth no necesitan instalarla.

La separación es importante porque cada capa responde una pregunta distinta:

| Capa | Pregunta |
|---|---|
| Proveedor | ¿De dónde vienen los datos? |
| Datos | ¿Qué estructura mínima representa una compañía? |
| Métricas | ¿Qué variables financieras se pueden comparar? |
| Scoring | ¿Cómo se traduce una tesis a ranking? |
| Selector | ¿Cómo se ejecuta el flujo completo? |

## Arquitectura De Correlación

La ruta de correlación vive en `CorrelationPortfolioSelector`.

```text
rank_within_groups
  -> _download_prices
  -> _apply_coverage_filter
  -> pct_change
  -> returns.corr
  -> _compute_corr_scores
  -> ranking_by_group
```

Después, `build_multigroup_portfolio` consolida candidatos por grupo y calcula
una selección final. `update_portfolio` usa la misma intuición para sugerir
adiciones de baja correlación frente a un portafolio existente.

## Diagramas

Los diagramas existentes siguen siendo la vista visual canónica:

| Diagrama | Uso |
|---|---|
| `diagrams/selection_module_architecture.drawio` | Vista universo del módulo. |
| `diagrams/selection_class_architecture.drawio` | Vista de clases, funciones y relaciones. |
| `diagrams/selection_workflow.drawio` | Vista orgánica del uso. |

Los enlaces editables están en `diagrams/open_in_diagrams_net.md`.

La política general de diagramas vive en `../_meta/04_diagram_policy.md`.

## Handoff A Otros Módulos

La salida más portable de `selection` es:

```python
selected_tickers = selected["ticker"].tolist()
```

Ese contrato puede alimentar:

- `optimization`, para asignación de pesos.
- `backtesting`, para simular estrategias.
- `risk`, para reportar exposición y pérdidas.
- notebooks, para investigación y análisis manual.
