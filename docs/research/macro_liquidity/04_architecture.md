# Arquitectura De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Pipeline De Construccion Del Analisis
4. Componentes Por Archivo
5. Relaciones Entre Clases
6. Puntos De Extension
7. Diagramas

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

Este documento muestra como se construye el analisis macroeconomico en codigo.
La arquitectura es un pipeline lineal con un cruce posterior hacia cliente y
selection.

## Pipeline De Construccion Del Analisis

```text
1. Catalogo
   default_us_macro_liquidity_catalog()
   -> tuple[EconomicSeriesSpec, ...]

2. Fuente de datos
   LocalSeriesProvider(frame).fetch(specs, start, end)
   o FredApiProvider(config).fetch(fred_specs, start, end)
   -> DataFrame(date, series, value)

3. Normalizacion
   normalize_series_frame(frame)
   -> date datetime, series str, value numeric

4. Panel de indicadores
   build_indicator_panel(series_frame, specs, min_periods)
   -> signal, zscore, signed_zscore, metadata del catalogo

5. Scores
   build_score_frame(indicators)
   -> block scores, macro_score, liquidity_score, coverage

6. Snapshot actual
   MacroLiquidityResearch.analyze_current(series_frame)
   -> MacroLiquidityResult(indicators, scores, current)

7. Forecast esperado
   RegimeForecast(scenarios=(ForecastScenario(...), ...))
   -> expected_macro_regime, expected_liquidity_regime

8. Vista de activos
   build_asset_class_view(current, forecast)
   -> AssetClassView

9. Overlay cliente
   build_selection_context(current, forecast, asset_view, policy)
   -> SelectionContext
```

## Componentes Por Archivo

| Archivo | Responsabilidad |
|---|---|
| `catalog.py` | Define `EconomicSeriesSpec` y catalogo base. |
| `providers.py` | Credenciales, providers, normalizacion inicial. |
| `transforms.py` | Transformacion de series y agregacion de scores. |
| `regimes.py` | Clasificacion de regimen actual. |
| `forecast.py` | Escenarios esperados in-house. |
| `asset_view.py` | Traduccion regimen-activos. |
| `src/client_policy/profiles.py` | Politica del cliente separada del research. |
| `src/strategy/policy_mapper.py` | Traduccion final a `SelectionContext`. |
| `src/strategy/selection_context.py` | Contrato consumible por selection. |

## Relaciones Entre Clases

```text
EconomicSeriesSpec
  -> usado por LocalSeriesProvider/FredApiProvider
  -> usado por build_indicator_panel

ProviderConfig
  -> usado por FredApiProvider para pedir FRED_API_KEY

MacroLiquidityResearch
  -> usa build_indicator_panel
  -> usa build_score_frame
  -> produce MacroLiquidityResult

MacroLiquidityResult.current
  -> entra a build_asset_class_view
  -> entra a build_selection_context

RegimeForecast
  -> entra a build_asset_class_view
  -> entra a build_selection_context

AssetClassView + ClientPolicy
  -> build_selection_context
  -> SelectionContext
```

## Puntos De Extension

| Extension | Donde Se Agrega | Cuidado |
|---|---|---|
| Nuevo indicador | `default_us_macro_liquidity_catalog` | Definir `higher_is_better`, `transform`, `periods`, `weight`. |
| Nuevo provider | `providers.py` | Debe cumplir `EconomicSeriesProvider.fetch`. |
| Alineacion de frecuencia | Antes de `build_indicator_panel` o dentro de nueva capa | Evitar mezclar fechas diarias, semanales, mensuales y trimestrales sin control. |
| Nueva matriz regimen-activos | `_STANCE_MATRIX`, `_STANCE_DETAILS` | Mantener trazabilidad de cambios. |
| Nuevo cliente | `src/client_policy` | No contaminar diagnostico economico. |

## Diagramas

- [macro_liquidity_module_architecture.drawio](./diagrams/macro_liquidity_module_architecture.drawio):
  vista de capas y handoff.
- [macro_liquidity_class_architecture.drawio](./diagrams/macro_liquidity_class_architecture.drawio):
  vista de clases y relaciones.
- [macro_liquidity_workflow.drawio](./diagrams/macro_liquidity_workflow.drawio):
  vista operativa para notebook.

Los enlaces editables estan en
[diagrams/open_in_diagrams_net.md](./diagrams/open_in_diagrams_net.md).
