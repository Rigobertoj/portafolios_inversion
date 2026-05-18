# Matriz De Capacidades De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Capacidades Principales
4. Capacidades De Providers
5. Capacidades De Transformacion Y Regimen
6. Capacidades De Forecast Y Vista De Activos
7. Capacidades De Cliente Y Handoff

## Mapa De La Serie

| Orden | Documento | Rol |
|---:|---|---|
| 00 | `00_index.md` | Entrada secuencial. |
| 01 | `01_audit_snapshot.md` | Estado real. |
| 02 | `02_conceptual_model.md` | Modelo mental. |
| 03 | `03_module_boundary.md` | Frontera. |
| 04 | `04_architecture.md` | Arquitectura. |
| 05 | `05_capability_matrix.md` | Capacidades. |
| 06 | `06_contracts.md` | Contratos. |
| 07 | `07_workflows.md` | Workflows. |
| 08 | `08_public_api.md` | API publica. |
| 09 | `09_classes_and_methods.md` | Clases y metodos. |
| 10 | `10_examples.md` | Ejemplos. |
| 11 | `11_validation_and_edge_cases.md` | Validacion. |
| 12 | `12_operational_notes.md` | Operacion. |
| 13 | `13_glossary.md` | Glosario. |

## Proposito

Esta matriz responde que objeto usar, que parametros recibe, que devuelve, que
errores puede levantar y para que caso de uso existe.

## Capacidades Principales

| Objeto | Tipo | Parametros | Retorna | Excepciones | Estado | Caso De Uso |
|---|---|---|---|---|---|---|
| `default_us_macro_liquidity_catalog` | Funcion | ninguno | `tuple[EconomicSeriesSpec, ...]` | ninguna esperada | Implementado | Obtener catalogo base US macro/liquidez. |
| `ProviderConfig` | Clase | `env=None`, `connection_specs=...` | instancia | ninguna en constructor | Implementado | Resolver credenciales fuera del repo. |
| `LocalSeriesProvider` | Clase | `frame: DataFrame` | instancia | `ValueError` si faltan columnas | Implementado | Usar datos locales/cacheados. |
| `FredApiProvider` | Clase | `config=None`, `timeout=30` | instancia | ninguna en constructor | Implementado | Descargar series FRED. |
| `MacroLiquidityResearch` | Clase | `series_specs=None`, `min_periods=6` | instancia | ninguna en constructor | Implementado | Orquestar diagnostico actual. |
| `ForecastScenario` | Dataclass | regimenes, probabilidad, horizonte, confianza | objeto escenario | ninguna | Implementado | Definir escenario in-house. |
| `RegimeForecast` | Dataclass | `scenarios`, `method`, `as_of` | objeto forecast | `ValueError` sin escenarios | Implementado | Agrupar escenarios y base case. |
| `build_asset_class_view` | Funcion | `current`, `forecast`, pesos, veto | `AssetClassView` | ninguna explicita | Implementado | Traducir regimenes a postura de activos. |
| `build_selection_context` | Funcion | `current`, `forecast`, `asset_view`, `policy` | `SelectionContext` | ninguna explicita | Implementado | Entregar contexto final a selection. |

## Capacidades De Providers

| Provider | Parametros De Uso | Retorna | Estado | Nota |
|---|---|---|---|---|
| `ProviderConfig.availability_report()` | ninguno | `DataFrame` | Implementado | Reporta credenciales, no descarga. |
| `ProviderConfig.require_token(provider)` | `provider: str` | `str` token | Implementado | Levanta `MissingCredentialError` si falta. |
| `LocalSeriesProvider.fetch(specs, start=None, end=None)` | specs y rango opcional | `DataFrame(date, series, value)` | Implementado | Filtra por nombres internos del catalogo. |
| `FredApiProvider.fetch(specs, start=None, end=None)` | specs FRED y rango opcional | `DataFrame(date, series, value)` | Implementado | Descarga solo specs con `provider="FRED"`. |
| BEA/BLS/Census/etc. | no hay clase | no aplica | Parcial | Solo existe metadata en `ApiConnectionSpec`. |

## Capacidades De Transformacion Y Regimen

| Funcion / Clase | Input | Output | Estado | Nota |
|---|---|---|---|---|
| `normalize_series_frame` | `DataFrame` | `DataFrame` normalizado | Implementado | Requiere `date`, `series`, `value`. |
| `build_indicator_panel` | series + specs + `min_periods` | panel largo con `signal`, `zscore`, `signed_zscore` | Implementado | No resamplea frecuencias. |
| `build_score_frame` | panel de indicadores | scores por bloque/engine | Implementado | Agrupa por fecha exacta. |
| `classify_macro_regime` | fila de score | etiqueta macro | Implementado | Heuristica inicial. |
| `classify_liquidity_regime` | score liquidez | etiqueta liquidez | Implementado | Umbrales fijos. |
| `MacroLiquidityResearch.analyze_current` | `series_frame` | `MacroLiquidityResult` | Implementado | Orquesta panel, scores y snapshot. |

## Capacidades De Forecast Y Vista De Activos

| Objeto | Input | Output | Estado | Nota |
|---|---|---|---|---|
| `ForecastScenario.normalized_probability(total)` | total de probabilidades | probabilidad normalizada | Implementado | Probabilidades negativas se truncan en 0. |
| `RegimeForecast.base_case` | escenarios | escenario de mayor probabilidad | Implementado | No pondera matriz de activos por escenarios. |
| `RegimeForecast.confidence` | escenarios | confianza ponderada | Implementado | Confianza no es probabilidad. |
| `RegimeForecast.scenario_table()` | ninguno | `DataFrame` de escenarios | Implementado | Para notebook/reporting. |
| `AssetClassView.to_frame()` | ninguno | `DataFrame` de una fila | Implementado | Para notebook/reporting. |

## Capacidades De Cliente Y Handoff

| Objeto | Input | Output | Estado | Nota |
|---|---|---|---|---|
| `mexican_moderate_aggressive_growth_policy()` | ninguno | `ClientPolicy` | Implementado | Perfil especifico actual. |
| `build_selection_context` | current, forecast, asset_view, policy | `SelectionContext` | Implementado | Cruza macro/liquidez con IPS. |
| `SelectionContext.as_dict()` | ninguno | `dict` | Implementado | Para integraciones. |
| `SelectionContext.to_frame()` | ninguno | `DataFrame` | Implementado | Para notebook/reporting. |
