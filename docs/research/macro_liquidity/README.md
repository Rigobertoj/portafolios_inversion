# Macro Liquidity Research

## Indice Del Documento

1. Que Es
2. Que Leer Primero
3. Mapa De La Serie
4. Estado Operativo En Una Frase
5. Salida Principal

## Que Es

`macro_liquidity` es la linea de codigo que convierte datos macroeconomicos y
de liquidez en una postura de research previa a seleccion. Su salida no es una
lista de tickers; es un `SelectionContext` con regimen actual, regimen esperado,
tilts, restricciones y racional para que `selection` pueda evaluar activos bajo
un contexto top-down.

## Que Leer Primero

Empieza en [00_index.md](./00_index.md). Si tu duda es "que esta realmente
implementado?", lee primero [01_audit_snapshot.md](./01_audit_snapshot.md).
Si tu duda es "como corre el flujo?", ve directo a
[07_workflows.md](./07_workflows.md).

## Mapa De La Serie

| Orden | Documento | Responde |
|---:|---|---|
| 00 | [00_index.md](./00_index.md) | Que leer segun la necesidad. |
| 01 | [01_audit_snapshot.md](./01_audit_snapshot.md) | Que existe, que es parcial y que riesgos hay. |
| 02 | [02_conceptual_model.md](./02_conceptual_model.md) | Cual es el modelo mental del modulo. |
| 03 | [03_module_boundary.md](./03_module_boundary.md) | Que hace, que no hace y con que se integra. |
| 04 | [04_architecture.md](./04_architecture.md) | Como se conectan archivos, clases y datos. |
| 05 | [05_capability_matrix.md](./05_capability_matrix.md) | Que clases/funciones existen y como se usan. |
| 06 | [06_contracts.md](./06_contracts.md) | Que contratos de datos, providers y outputs sostiene. |
| 07 | [07_workflows.md](./07_workflows.md) | Como ejecutar descarga, cache local y flujo completo. |
| 08 | [08_public_api.md](./08_public_api.md) | Que imports publicos expone el paquete. |
| 09 | [09_classes_and_methods.md](./09_classes_and_methods.md) | Parametros, retornos, errores y relaciones por clase/metodo. |
| 10 | [10_examples.md](./10_examples.md) | Ejemplos minimos copiables para notebook. |
| 11 | [11_validation_and_edge_cases.md](./11_validation_and_edge_cases.md) | Como validar y que limites vigilar. |
| 12 | [12_operational_notes.md](./12_operational_notes.md) | Credenciales, ambiente, cache y operacion. |
| 13 | [13_glossary.md](./13_glossary.md) | Vocabulario del modulo. |
| 18 | [18_policy_workflow_diagram.md](./18_policy_workflow_diagram.md) | Que modifica `MacroLiquidityPolicy` y como fluye hasta selection. |

Los articulos `14`, `15` y `16` son complementarios: profundizan en regimen
actual, regimen esperado/vista de activos y handoff cliente-selection.

## Estado Operativo En Una Frase

Hoy el modulo puede correr end-to-end con datos locales y puede descargar FRED
si existe `FRED_API_KEY`; el resto de fuentes publicas estan registradas como
metadata de conexion, pero no tienen downloader operativo todavia.

## Salida Principal

La salida final del flujo es `SelectionContext`, definido en
`src/strategy/selection_context.py`. Ese objeto transporta hacia `selection`:
regimenes, moneda base, perfil de riesgo, asset-class tilts, estilos preferidos,
exclusiones, activos elegibles, controles de riesgo, score tilts y racional.
