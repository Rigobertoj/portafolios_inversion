# Contratos De Datos E Integracion De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Contrato `EconomicSeriesSpec`
4. Contrato `SeriesFrame`
5. Contrato `IndicatorPanel`
6. Contrato `ScoreFrame`
7. Contratos De Objetos De Dominio
8. Contrato De Providers
9. Contrato Final Hacia Selection
10. Condiciones De Error

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

Este documento define que recibe y entrega cada etapa. Si un modulo externo se
conecta con `macro_liquidity`, debe respetar estos contratos.

## Contrato `EconomicSeriesSpec`

| Campo | Tipo | Requerido | Uso |
|---|---|---:|---|
| `name` | `str` | Si | Nombre interno usado en `series`. |
| `engine` | `"macro"` o `"liquidity"` | Si | Separa macro tradicional de liquidez. |
| `block` | `str` | Si | Grupo de score: growth, inflation, credit, etc. |
| `higher_is_better` | `bool` | Si | Direccion economica del z-score. |
| `provider` | `str` | No | Fuente externa esperada. |
| `provider_code` | `str` | No | Codigo de la serie en el provider. |
| `frequency` | `str` | No | Frecuencia informativa. |
| `transform` | `level/diff/pct_change/yoy_change` | No | Transformacion antes de z-score. |
| `periods` | `int` | No | Periodos de la transformacion. |
| `weight` | `float` | No | Peso en agregacion. |
| `required` | `bool` | No | Indicador critico. |
| `description` | `str` | No | Nota humana. |

## Contrato `SeriesFrame`

Entrada minima para cualquier provider:

| Campo | Tipo Esperado | Validacion |
|---|---|---|
| `date` | convertible a `datetime64` | No nulo despues de conversion. |
| `series` | `str` | Debe coincidir con `EconomicSeriesSpec.name`. |
| `value` | numerico | Se convierte con `pd.to_numeric`; nulos se eliminan. |

`normalize_series_frame` elimina filas invalidas y ordena por `series`, `date`.

## Contrato `IndicatorPanel`

`build_indicator_panel` agrega:

| Campo | Origen |
|---|---|
| `engine`, `block`, `higher_is_better`, `weight`, `required` | Catalogo. |
| `provider`, `provider_code` | Catalogo. |
| `signal` | Transformacion de `value`. |
| `zscore` | Z-score expansivo de `signal`. |
| `signed_zscore` | `zscore` con direccion economica. |

## Contrato `ScoreFrame`

`build_score_frame` produce una tabla por fecha:

| Campo | Significado |
|---|---|
| `<block>_score` | Score por bloque economico. |
| `macro_score` | Score agregado de engine macro. |
| `liquidity_score` | Score agregado de engine liquidity. |
| `macro_coverage` | Proporcion de senales macro disponibles. |
| `liquidity_coverage` | Proporcion de senales liquidity disponibles. |

Riesgo actual: la agregacion es por fecha exacta, no por fecha macro alineada.

## Contratos De Objetos De Dominio

| Objeto | Campos Principales | Productor | Consumidor |
|---|---|---|---|
| `CurrentRegimeSnapshot` | fecha, regimenes, scores, coverage, confidence | `MacroLiquidityResearch` | asset view, strategy |
| `ForecastScenario` | name, regimenes, probability, horizon, confidence | research in-house | `RegimeForecast` |
| `RegimeForecast` | scenarios, method, as_of | research in-house | asset view, strategy |
| `AssetClassView` | stances, tilts, estilos, controles, rationale | `build_asset_class_view` | strategy |
| `ClientPolicy` | perfil + mandato | `client_policy` | strategy |
| `SelectionContext` | regimenes, tilts, filtros, controles, rationale | strategy | selection |

## Contrato De Providers

Todo provider operativo debe implementar:

```python
fetch(specs: Iterable[EconomicSeriesSpec], start: str | None, end: str | None) -> pd.DataFrame
```

Y devolver `date`, `series`, `value`.

Hoy cumplen esto:

- `LocalSeriesProvider`
- `FredApiProvider`

No cumplen todavia, porque no existen como clases operativas:

- BEA
- BLS
- Census
- Treasury
- Banxico
- INEGI
- EIA
- BIS
- NY Fed
- Fed DDP

## Contrato Final Hacia Selection

`SelectionContext` entrega:

| Campo | Uso Esperado |
|---|---|
| `asset_class_overweights` | Universos a favorecer. |
| `asset_class_underweights` | Universos a penalizar o revisar. |
| `preferred_styles` | Estilos/factores para scoring. |
| `excluded_sectors` | Filtros por IPS. |
| `eligible_assets` | Tipos de activo permitidos. |
| `risk_controls` | Validaciones antes/despues de selection. |
| `score_tilts` | Ajustes numericos de scoring. |
| `rationale` | Auditoria del contexto. |

## Condiciones De Error

| Error | Origen | Causa |
|---|---|---|
| `ValueError` | `normalize_series_frame` | Faltan columnas requeridas. |
| `MissingCredentialError` | `ProviderConfig.require_token` | Falta token requerido. |
| `requests.HTTPError` | `FredApiProvider._fetch_one` | Falla HTTP de FRED. |
| `ValueError` | `MacroLiquidityResearch._snapshot_from_scores` | Scores vacios. |
| `ValueError` | `RegimeForecast.__post_init__` | No hay escenarios. |
