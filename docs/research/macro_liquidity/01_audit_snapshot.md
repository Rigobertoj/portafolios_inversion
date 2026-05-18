# Audit Snapshot De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Resumen Ejecutivo
4. Estado De Capacidades
5. Estado De APIs Y Fuentes
6. Riesgos Actuales
7. Siguientes Acciones Naturales

## Mapa De La Serie

| Orden | Documento | Rol |
|---:|---|---|
| 00 | `00_index.md` | Entrada secuencial. |
| 01 | `01_audit_snapshot.md` | Estado real del modulo. |
| 02 | `02_conceptual_model.md` | Modelo mental. |
| 03 | `03_module_boundary.md` | Frontera e integracion. |
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

Este documento dice que existe hoy en codigo y que no. Es el primer documento a
leer para evitar confundir diseno deseado con funcionalidad implementada.

## Resumen Ejecutivo

El flujo macro/liquidez esta implementado como pipeline local de research:

```text
catalogo -> provider/local data -> indicadores -> scores -> regimen actual
-> forecast in-house -> vista de activos -> overlay cliente -> SelectionContext
```

La descarga real por API esta implementada solo para FRED mediante
`FredApiProvider`. Las demas fuentes publicas estan descritas en
`ApiConnectionSpec`, pero todavia no tienen clase provider operativa.

## Estado De Capacidades

| Elemento | Estado | Evidencia | Riesgo / Siguiente accion |
|---|---|---|---|
| Catalogo US macro/liquidez | Implementado | `default_us_macro_liquidity_catalog()` | Revisar pesos y series antes de usar en produccion. |
| Contrato de serie larga | Implementado | `normalize_series_frame()` | Requiere columnas `date`, `series`, `value`. |
| Provider local/cache | Implementado | `LocalSeriesProvider.fetch()` | Depende de que el CSV/DataFrame ya venga correcto. |
| Descarga FRED | Implementado | `FredApiProvider.fetch()` | Requiere `FRED_API_KEY`; ignora specs no FRED. |
| Metadata BEA/BLS/Census/Treasury/etc. | Parcial | `default_api_connection_specs()` | Hay metadata, pero no downloader por fuente. |
| Diagnostico regimen actual | Implementado | `MacroLiquidityResearch.analyze_current()` | No alinea frecuencias a una fecha comun. |
| Forecast in-house | Implementado | `ForecastScenario`, `RegimeForecast` | No estima escenarios automaticamente. |
| Vista top-down de activos | Implementado | `build_asset_class_view()` | Matriz de stances inicial, no calibrada por backtest. |
| Politica cliente | Implementado | `ClientPolicy`, helper mexicano | Solo hay un helper de cliente especifico. |
| Handoff a selection | Implementado | `build_selection_context()` | Selection debe saber consumir estos campos. |
| Notebook workflow | Implementado | `research/12.20260508.macro_liquidity_workflow.ipynb` | Depende de datos disponibles o credenciales. |

## Estado De APIs Y Fuentes

| Fuente | Estado En Codigo | Variable | Puede Descargar Hoy Desde Este Modulo |
|---|---|---|---|
| FRED | Provider operativo | `FRED_API_KEY` | Si, con `FredApiProvider`. |
| BEA | Metadata | `BEA_API_KEY` | No, falta provider. |
| BLS | Metadata | `BLS_API_KEY` | No, falta provider. |
| Census | Metadata | `CENSUS_API_KEY` | No, falta provider. |
| Fed DDP | Metadata | Ninguna | No, falta provider. |
| NY Fed | Metadata | Ninguna | No, falta provider. |
| Treasury Fiscal Data | Metadata | Ninguna | No, falta provider. |
| EIA | Metadata | `EIA_API_KEY` | No, falta provider. |
| BIS | Metadata | Ninguna | No, falta provider. |
| INEGI | Metadata | `INEGI_TOKEN` | No, falta provider. |
| Banxico | Metadata y una serie en catalogo | `BANXICO_TOKEN` | No, falta provider. |

## Riesgos Actuales

- `availability_report()` indica disponibilidad de credenciales, no existencia
  de downloader.
- El ultimo snapshot puede reflejar fechas de series de alta frecuencia porque
  no existe una etapa de alineacion mensual/trimestral del panel.
- `confidence` es cobertura de senales, no probabilidad de estar en lo correcto.
- El forecast esperado es manual; el modulo no entrena ni estima escenarios.
- No hay persistencia/caching automatica de descargas FRED.

## Siguientes Acciones Naturales

1. Agregar `PanelAlignment` o transformacion mensual antes de scorear.
2. Implementar providers adicionales empezando por Banxico/Treasury/BEA.
3. Agregar cache reproducible para descargas.
4. Agregar tests para frecuencia mixta y missing required series.
5. Conectar `SelectionContext` de forma explicita dentro de `selection`.
