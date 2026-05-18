# Glosario De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Terminos De Estado
4. Terminos De Datos
5. Terminos De Regimen
6. Terminos De Handoff

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

Fijar vocabulario para que research, cliente, selection y documentacion usen el
mismo lenguaje.

## Terminos De Estado

| Termino | Definicion |
|---|---|
| Implementado | Existe codigo funcional y evidencia local. |
| Parcial | Existe interfaz, metadata o estructura, pero falta funcionalidad operativa. |
| Planeado | Existe como intencion/documento, no como codigo. |
| Externo | Depende de credenciales, APIs, servicios o datos fuera del repo. |
| Riesgo | Corre, pero tiene limitacion tecnica, metodologica u operativa. |

## Terminos De Datos

| Termino | Definicion |
|---|---|
| `EconomicSeriesSpec` | Metadata de una serie economica. |
| `SeriesFrame` | `DataFrame` largo con `date`, `series`, `value`. |
| `IndicatorPanel` | Panel con senales, z-scores y metadata economica. |
| `ScoreFrame` | Tabla agregada con scores por bloque y engine. |
| Provider | Clase que entrega `SeriesFrame`. |
| Cache local | Dataset guardado para correr sin API. |

## Terminos De Regimen

| Termino | Definicion |
|---|---|
| Regimen macro | Etiqueta del estado economico tradicional. |
| Regimen de liquidez | Etiqueta de condiciones de liquidez/financieras. |
| `CurrentRegimeSnapshot` | Foto actual de regimen y scores. |
| `RegimeForecast` | Escenarios esperados definidos in-house. |
| `AssetClassView` | Postura top-down antes del cliente. |
| Liquidity stress veto | Regla que prioriza cautela si la liquidez actual es `estresada`. |

## Terminos De Handoff

| Termino | Definicion |
|---|---|
| `ClientPolicy` | Perfil del cliente mas mandato IPS. |
| `SelectionContext` | Contrato final que recibe selection. |
| `score_tilts` | Ajustes numericos para sesgar scoring. |
| `risk_controls` | Controles de riesgo derivados de macro/liquidez y cliente. |
| Handoff | Punto donde un modulo entrega una salida consumible por otro. |
