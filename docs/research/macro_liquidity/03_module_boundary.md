# Frontera Del Modulo `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Responsabilidad Principal
4. Dentro Del Modulo
5. Fuera Del Modulo
6. Dependencias Entrantes
7. Dependencias Salientes
8. Contrato De Integracion Principal

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

Este documento define exactamente que responsabilidad tiene la linea de research
macro/liquidez y donde termina. Es la pieza que evita mezclar macro research,
perfil de cliente, seleccion de activos y optimizacion.

## Responsabilidad Principal

`src/research/macro_liquidity` toma series economicas normalizadas y produce un
diagnostico de regimen actual junto con una vista top-down de activos basada en
un forecast in-house.

## Dentro Del Modulo

- Definir metadata de indicadores con `EconomicSeriesSpec`.
- Resolver credenciales con `ProviderConfig`.
- Descargar FRED con `FredApiProvider`.
- Leer datos locales/cacheados con `LocalSeriesProvider`.
- Normalizar `DataFrame(date, series, value)`.
- Transformar series a senales comparables.
- Agregar scores por bloque y engine.
- Clasificar regimen macro y regimen de liquidez.
- Representar escenarios de forecast in-house.
- Traducir regimenes a `AssetClassView`.

## Fuera Del Modulo

- Descargar BEA, BLS, Banxico, INEGI, Treasury u otras fuentes todavia no
  implementadas.
- Estimar automaticamente probabilidades de forecast.
- Seleccionar tickers o ETFs concretos.
- Optimizar pesos.
- Hacer backtesting de cartera final.
- Aplicar restricciones del cliente dentro del diagnostico economico.
- Guardar API keys dentro del repositorio.

## Dependencias Entrantes

| Productor | Entra Como | Consumidor |
|---|---|---|
| FRED API | Observaciones JSON | `FredApiProvider` |
| CSV/DataFrame local | `date`, `series`, `value` | `LocalSeriesProvider` |
| Research in-house | `ForecastScenario` | `RegimeForecast` |
| IPS/cliente | `ClientPolicy` | `build_selection_context` |

## Dependencias Salientes

| Salida | Consumidor | Proposito |
|---|---|---|
| `MacroLiquidityResult` | Notebook/research | Auditar indicadores, scores y snapshot. |
| `CurrentRegimeSnapshot` | `build_asset_class_view` | Postura actual. |
| `AssetClassView` | `build_selection_context` | Tilts top-down antes de cliente. |
| `SelectionContext` | `selection` | Contexto final para filtrar y scorear activos. |

## Contrato De Integracion Principal

El contrato final es:

```text
CurrentRegimeSnapshot + RegimeForecast + AssetClassView + ClientPolicy
-> SelectionContext
```

`SelectionContext` vive en `src/strategy`, no en `src/research`, porque ya es la
frontera entre research economico, politica del cliente y seleccion.
