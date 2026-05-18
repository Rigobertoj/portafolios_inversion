# Validacion Y Casos Limite De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Validaciones Del Codigo Actual
4. Validaciones Antes De Research
5. Casos Limite
6. Gaps Tecnicos Y Metodologicos
7. Comandos Recomendados

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

Este documento distingue entre pruebas existentes, validaciones que debe hacer
el investigador y limites que el codigo todavia no resuelve.

## Validaciones Del Codigo Actual

`test/research/test_macro_liquidity_research.py` valida:

- Construccion de `MacroLiquidityResearch` con specs sinteticos.
- Generacion de `CurrentRegimeSnapshot`.
- Construccion de `RegimeForecast`.
- Construccion de `AssetClassView`.
- Aplicacion del helper de cliente mexicano.
- Generacion de `SelectionContext`.
- `ProviderConfig.availability_report()`.
- `MissingCredentialError` cuando falta FRED.

## Validaciones Antes De Research

Antes de interpretar resultados:

- Revisar `ProviderConfig().availability_report()`.
- Confirmar que el dataset tiene `date`, `series`, `value`.
- Confirmar que `series` coincide con `EconomicSeriesSpec.name`.
- Revisar fecha maxima por serie.
- Revisar `result.scores.tail()`.
- Revisar `macro_coverage` y `liquidity_coverage`.
- Documentar racional de cada `ForecastScenario`.

## Casos Limite

| Caso | Comportamiento | Riesgo |
|---|---|---|
| Faltan columnas en datos | `ValueError` | El workflow no inicia. |
| Falta FRED key | `MissingCredentialError` | No descarga FRED. |
| FRED HTTP error | `requests.HTTPError` | Descarga incompleta. |
| No hay specs FRED | `FredApiProvider.fetch` devuelve DataFrame vacio | Snapshot falla despues. |
| Scores vacios | `ValueError` | No hay regimen actual. |
| Forecast sin escenarios | `ValueError` | No hay base case. |
| Liquidez `estresada` | Veto usa current stance | Puede ignorar forecast favorable. |
| Frecuencias mezcladas | El codigo corre | Interpretacion puede ser fragil. |

## Gaps Tecnicos Y Metodologicos

| Gap | Impacto |
|---|---|
| No hay alineacion de frecuencia | El ultimo score puede no representar un corte macro consistente. |
| No hay providers BEA/BLS/Banxico/etc. | Solo FRED y cache local son operativos. |
| No hay cache automatico | Reproducibilidad depende del usuario. |
| No hay versionado de dataset | Dificulta auditoria historica del research. |
| No hay calibracion de `_STANCE_MATRIX` | La vista de activos es heuristica inicial. |
| No hay integracion concreta en `selection` | `SelectionContext` existe, pero debe ser consumido por selection. |

## Comandos Recomendados

```bash
pytest test/research/test_macro_liquidity_research.py
```

Validar docs:

```bash
find docs/research/macro_liquidity -name '*.md' -print | sort
find docs/research/macro_liquidity -name '*.md' -print | while read file; do
  grep -q 'Indice Del Documento' "$file" || echo "$file"
done
```
