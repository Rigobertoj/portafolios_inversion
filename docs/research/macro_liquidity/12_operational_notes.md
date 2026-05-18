# Notas Operacionales De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Variables De Entorno
4. Configuracion De FRED
5. Modo Cache Local
6. Manejo De Errores
7. Reproducibilidad
8. Mantenimiento

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

Este documento explica como operar el modulo fuera de la teoria: credenciales,
ambiente, fallas esperadas y reproducibilidad.

## Variables De Entorno

| Variable | Uso Actual |
|---|---|
| `FRED_API_KEY` | Usada por `FredApiProvider`. |
| `BEA_API_KEY` | Registrada en metadata, sin provider operativo. |
| `BLS_API_KEY` | Registrada en metadata, sin provider operativo. |
| `CENSUS_API_KEY` | Registrada en metadata, sin provider operativo. |
| `EIA_API_KEY` | Registrada en metadata, sin provider operativo. |
| `INEGI_TOKEN` | Registrada en metadata, sin provider operativo. |
| `BANXICO_TOKEN` | Registrada en metadata, sin provider operativo. |

Las credenciales nunca deben guardarse en el repo.

## Configuracion De FRED

Linux/WSL:

```bash
export FRED_API_KEY="tu_key"
```

PowerShell:

```powershell
$env:FRED_API_KEY="tu_key"
```

Validacion:

```python
from src.research.macro_liquidity import ProviderConfig

ProviderConfig().is_available("FRED")
ProviderConfig().availability_report()
```

## Modo Cache Local

El modo mas reproducible es guardar un dataset local con:

```text
date,series,value
```

y cargarlo con `LocalSeriesProvider`.

Recomendacion operativa: guardar tambien metadata externa del dataset, por
ejemplo fecha de descarga, fuente, rango temporal y version del catalogo usado.

## Manejo De Errores

| Error | Accion |
|---|---|
| `MissingCredentialError` | Configurar variable de entorno o usar cache local. |
| `requests.HTTPError` | Revisar FRED, serie, key, red y rango de fechas. |
| `ValueError` por columnas | Corregir CSV/DataFrame al contrato `SeriesFrame`. |
| `ValueError` por forecast vacio | Agregar al menos un `ForecastScenario`. |

## Reproducibilidad

Para reportes serios:

- Congelar `series_data` a CSV/parquet.
- Guardar `forecast.scenario_table()`.
- Guardar `selection_context.as_dict()`.
- Registrar fecha de ejecucion y rango de datos.
- Registrar si se uso API o cache local.

## Mantenimiento

Cuando se agregue un provider nuevo:

1. Implementar clase con contrato `fetch(specs, start, end)`.
2. Agregar tests sin depender de red real.
3. Actualizar `01_audit_snapshot.md`.
4. Actualizar `05_capability_matrix.md`.
5. Actualizar `07_workflows.md`.
6. Actualizar `12_operational_notes.md`.
