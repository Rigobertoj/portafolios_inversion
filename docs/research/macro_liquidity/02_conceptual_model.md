# Modelo Conceptual De `macro_liquidity`

## Indice Del Documento

1. Mapa De La Serie
2. Problema Que Resuelve
3. Idea Central
4. Separacion Entre Economia, Cliente Y Selection
5. Ruta Conceptual Del Analisis
6. Que No Debe Interpretarse De Mas

## Mapa De La Serie

| Orden | Documento | Rol |
|---:|---|---|
| 00 | `00_index.md` | Entrada secuencial. |
| 01 | `01_audit_snapshot.md` | Estado real: implementado, parcial, externo y riesgos. |
| 02 | `02_conceptual_model.md` | Modelo mental del analisis. |
| 03 | `03_module_boundary.md` | Frontera, dependencias y responsabilidades. |
| 04 | `04_architecture.md` | Componentes y flujo interno. |
| 05 | `05_capability_matrix.md` | Clases, funciones, parametros, retornos y casos de uso. |
| 06 | `06_contracts.md` | Contratos de datos, providers y outputs. |
| 07 | `07_workflows.md` | Ejecucion paso a paso. |
| 08 | `08_public_api.md` | API publica por paquete. |
| 09 | `09_classes_and_methods.md` | Detalle por clase y metodo. |
| 10 | `10_examples.md` | Ejemplos ejecutables. |
| 11 | `11_validation_and_edge_cases.md` | Validacion, gaps y limites. |
| 12 | `12_operational_notes.md` | Configuracion, credenciales y operacion. |
| 13 | `13_glossary.md` | Vocabulario. |

## Problema Que Resuelve

`macro_liquidity` construye una postura macroeconomica previa a seleccion. Su
objetivo no es elegir instrumentos, sino transformar series economicas en un
contexto util para decidir que tipos de activos, estilos y controles deben
favorecerse antes de entrar a `selection`.

La pregunta operativa es:

```text
En que regimen macroeconomico y de liquidez esta Estados Unidos,
que regimen esperamos en el futuro cercano,
y que contexto debe recibir selection para evaluar activos?
```

## Idea Central

El modulo separa dos momentos:

1. `CurrentRegimeSnapshot`: fotografia actual construida desde datos historicos.
2. `RegimeForecast`: escenario esperado definido in-house por research.

Despues combina ambos en `AssetClassView` y finalmente aplica `ClientPolicy`
para generar `SelectionContext`.

## Separacion Entre Economia, Cliente Y Selection

| Capa | Codigo | Que Decide | Que No Decide |
|---|---|---|---|
| Economia | `src/research/macro_liquidity` | Regimen actual, regimen esperado y vista top-down. | Restricciones del cliente o tickers. |
| Cliente | `src/client_policy` | Moneda base, horizonte, drawdown, universo elegible y exclusiones. | Regimen economico. |
| Handoff | `src/strategy` | Traduccion a `SelectionContext`. | Seleccion concreta de instrumentos. |
| Selection | `src/selection` | Candidatos concretos segun su propia logica. | Diagnostico macro o IPS. |

## Ruta Conceptual Del Analisis

```text
EconomicSeriesSpec
-> Provider fetch / LocalSeriesProvider
-> normalize_series_frame
-> build_indicator_panel
-> build_score_frame
-> MacroLiquidityResearch.analyze_current
-> CurrentRegimeSnapshot
-> RegimeForecast
-> build_asset_class_view
-> AssetClassView
-> build_selection_context
-> SelectionContext
```

Cada flecha tiene un contrato documentado en `06_contracts.md` y un objeto
responsable documentado en `05_capability_matrix.md`.

## Que No Debe Interpretarse De Mas

- `macro_score` y `liquidity_score` no son retornos esperados.
- `confidence` mide cobertura de senales, no probabilidad de acierto.
- `RegimeForecast` no pronostica automaticamente; recibe una tesis in-house.
- `AssetClassView` no contiene tickers.
- `ProviderConfig.availability_report()` no implica que exista downloader para
  cada fuente. Hoy solo `FredApiProvider` y `LocalSeriesProvider` descargan o
  entregan datos operativamente.
