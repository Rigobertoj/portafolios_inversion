# Selection: API Reference Narrativa

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. API Pública
4. Clases Principales
5. Funciones Principales
6. Atributos De Estado
7. Cómo Leer La Referencia Heredada

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

Esta referencia narrativa resume la API pública y explica cómo leerla. La
referencia heredada completa vive en `api_reference.md`.

## API Pública

`src.selection` expone:

- `FundamentalSelector`
- `CorrelationPortfolioSelector`
- `CorrelationSelector`
- `YahooFundamentalsProvider`
- `FundamentalData`
- `FundamentalScoreConfig`
- `MetricSignalSpec`
- `ValueScoreConfig`
- `GrowthScoreConfig`
- `FUNDAMENTAL_METRIC_SIGNAL_SPECS`
- `fundamental_metric_signal_specs`
- builders de métricas
- builders de paneles y targets forward
- scoring aprendido con XGBoost
- funciones de scoring

## Clases Principales

| Clase | Uso |
|---|---|
| `FundamentalSelector` | Orquestador de selección fundamental. |
| `YahooFundamentalsProvider` | Proveedor de datos fundamentales de Yahoo Finance. |
| `FundamentalData` | Registro de datos fundamentales por ticker. |
| `MetricSignalSpec` | Componente individual de score. |
| `FundamentalScoreConfig` | Configuración genérica de score. |
| `ValueScoreConfig` | Configuración default value. |
| `GrowthScoreConfig` | Configuración default growth. |
| `CorrelationPortfolioSelector` | Selector por baja correlación. |
| `FundamentalLearningPanelBuilder` | Constructor de paneles históricos para research. |
| `XGBoostFundamentalModel` | Modelo no lineal para aprender impacto de señales. |
| `LearnedScoreConfigFactory` | Convierte impactos en `FundamentalScoreConfig`. |
| `LearnedFundamentalSelector` | Selector con configs aprendidas por grupo. |

## Funciones Principales

| Función | Uso |
|---|---|
| `build_fundamental_metrics` | Construye métricas para un `FundamentalData`. |
| `build_metrics_frame` | Construye tabla de métricas para varios registros. |
| `build_fundamental_metric_history` | Construye historia de métricas por ticker. |
| `build_metric_history_frame` | Construye tabla histórica larga. |
| `build_forward_return_targets` | Alinea métricas históricas con retornos forward. |
| `build_fundamental_learning_panel` | Construye panel de señales y targets. |
| `candidate_feature_columns` | Lista features `metric__signal` útiles para modelos. |
| `score_fundamentals` | Calcula ranking fundamental actual. |
| `score_fundamentals_over_time` | Calcula ranking por periodo. |
| `fundamental_metric_signal_specs` | Genera specs para columnas fundamentales. |

## Atributos De Estado

### `FundamentalSelector`

| Atributo | Se Llena Cuando | Interpretación |
|---|---|---|
| `raw_data` | `collect_metrics`, `collect_metric_history`, `rank` | Datos fundamentales crudos. |
| `metrics_` | `collect_metrics`, `rank` | Métricas calculadas. |
| `metric_history_` | `collect_metric_history`, `rank_over_time`, `rank` con señales históricas | Métricas por periodo. |
| `ranking_` | `rank` | Ranking actual. |
| `ranking_history_` | `rank_over_time` | Ranking temporal. |

### `CorrelationPortfolioSelector`

| Atributo | Se Llena Cuando | Interpretación |
|---|---|---|
| `prices_by_group` | `rank_within_groups` | Precios usados por grupo. |
| `returns_by_group` | `rank_within_groups` | Retornos usados por grupo. |
| `ranking_by_group` | `rank_within_groups` | Ranking por menor correlación. |

### `XGBoostFundamentalModel`

| Atributo | Se Llena Cuando | Interpretación |
|---|---|---|
| `impact_report_` | `fit` | Importancia, dirección, estabilidad, cobertura e impacto ajustado por señal. |
| `training_summary_` | `fit` | Observaciones, features y diagnóstico fuera de muestra por grupo. |
| `models_` | `fit` | Estimadores ajustados por grupo. |

## Cómo Leer La Referencia Heredada

Usar `api_reference.md` para detalles de firmas, parámetros y retornos. Usar
esta referencia narrativa para interpretar qué pieza conviene llamar y por qué.

Regla práctica:

- Si quieres entender, lee `04`, `05` y `06`.
- Si quieres ejecutar, lee `07`.
- Si quieres consultar firma exacta, lee `api_reference.md`.
