# Indice Operativo De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Que Leer Segun La Necesidad
4. Ruta Minima Para Entender El Codigo
5. Evidencia Usada

## Proposito

Este indice ordena la documentacion del modulo de research macro/liquidez como
una ruta de lectura. La prioridad es que puedas ver rapidamente que esta
implementado, como se construye el analisis macroeconomico en codigo y como se
conecta con cliente y seleccion.

## Mapa De La Serie

| Orden | Documento | Rol | Cuando Leerlo |
|---:|---|---|---|
| 00 | `00_index.md` | Entrada secuencial | Siempre primero. |
| 01 | `01_audit_snapshot.md` | Estado real | Para distinguir implementado, parcial, externo y riesgos. |
| 02 | `02_conceptual_model.md` | Modelo mental | Para entender la logica economica general. |
| 03 | `03_module_boundary.md` | Frontera | Antes de integrar con otros modulos. |
| 04 | `04_architecture.md` | Arquitectura | Antes de modificar codigo. |
| 05 | `05_capability_matrix.md` | Inventario tecnico | Para saber que clase o funcion usar. |
| 06 | `06_contracts.md` | Contratos | Antes de conectar datos o APIs. |
| 07 | `07_workflows.md` | Uso operativo | Para ejecutar en notebook. |
| 08 | `08_public_api.md` | Imports publicos | Para usar el paquete desde fuera. |
| 09 | `09_classes_and_methods.md` | Referencia detallada | Para parametros, retornos y excepciones. |
| 10 | `10_examples.md` | Ejemplos | Para copiar un flujo minimo. |
| 11 | `11_validation_and_edge_cases.md` | Validacion | Antes de confiar en resultados. |
| 12 | `12_operational_notes.md` | Operacion | Para credenciales, caches y fallas esperadas. |
| 13 | `13_glossary.md` | Vocabulario | Cuando haya duda de terminos. |
| 17 | `17_confidence_forecast_scenario_policy.md` | Explicacion analitica | Para entender confianza, forecast y escenarios sin caja negra. |
| 18 | `18_policy_workflow_diagram.md` | Politica y diagramas | Para ver que modifica `MacroLiquidityPolicy` y como fluye hasta selection. |

## Que Leer Segun La Necesidad

| Necesidad | Documento |
|---|---|
| Saber si las APIs descargan hoy | `01_audit_snapshot.md` y `12_operational_notes.md` |
| Entender la construccion del analisis macro | `02_conceptual_model.md`, `04_architecture.md`, `07_workflows.md` |
| Ver parametros y retornos de clases | `05_capability_matrix.md` y `09_classes_and_methods.md` |
| Conectar datos externos | `06_contracts.md` y `12_operational_notes.md` |
| Usar el flujo en notebook | `07_workflows.md` y `10_examples.md` |
| Integrar con selection | `03_module_boundary.md`, `06_contracts.md`, `08_public_api.md` |
| Explicar confidence, forecast y escenarios | `17_confidence_forecast_scenario_policy.md` |
| Ver que puedes modificar en `MacroLiquidityPolicy` | `18_policy_workflow_diagram.md` |

## Ruta Minima Para Entender El Codigo

Lee en este orden:

1. `01_audit_snapshot.md`
2. `03_module_boundary.md`
3. `04_architecture.md`
4. `05_capability_matrix.md`
5. `07_workflows.md`
6. `09_classes_and_methods.md`

Con esa ruta puedes responder: que existe, como se alimenta, que transforma,
que objetos salen y donde entra `selection`.

## Evidencia Usada

La serie esta verificada contra:

- `src/research/macro_liquidity/*.py`
- `src/client_policy/*.py`
- `src/strategy/*.py`
- `test/research/test_macro_liquidity_research.py`
- `research/12.20260508.macro_liquidity_workflow.ipynb`
- `research/13.20260508.macro_liquidity.ipynb`
- `docs/research/macro_liquidity/diagrams/*.drawio`
