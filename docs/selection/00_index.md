# Selection: Índice Y Ruta De Lectura

## Índice Del Documento

1. Propósito
2. Lectura Recomendada
3. Mapa De La Serie
4. Relación Con Documentación Heredada
5. Qué Leer Según La Necesidad
6. Estado Del Piloto

## Mapa De La Serie

| Orden | Documento | Rol | Cuándo Leerlo |
|---:|---|---|---|
| 00 | `00_index.md` | Entrada secuencial | Siempre primero. |
| 01 | `01_conceptual_model.md` | Modelo conceptual | Antes de usar o modificar `selection`. |
| 02 | `02_architecture.md` | Arquitectura del módulo | Antes de tocar código. |
| 03 | `03_data_contracts.md` | Contratos de datos | Cuando se integran entradas o salidas. |
| 04 | `04_fundamental_selection_article.md` | Artículo fundamental | Para entender value, growth y scoring. |
| 05 | `05_correlation_selection_article.md` | Artículo correlación | Para entender diversificación estadística. |
| 06 | `06_scoring_model.md` | Modelo de score | Para ajustar métricas, señales o pesos. |
| 07 | `07_workflows.md` | Flujos prácticos | Para ejecutar casos de uso. |
| 08 | `08_api_reference.md` | API narrativa | Para consultar clases, métodos y atributos. |
| 09 | `09_validation_and_edge_cases.md` | Validación | Para pruebas, errores y límites. |
| 10 | `10_glossary.md` | Glosario | Para unificar vocabulario. |

## Propósito

Este documento es el punto de entrada de la documentación piloto del módulo
`src.selection`. Su objetivo es resolver la pregunta que motivó la
reestructura:

```text
Cuando entro a la documentación de selection, ¿qué leo primero y por qué?
```

La respuesta es: primero este índice, luego el modelo conceptual, luego la
arquitectura y después los artículos según la ruta que se quiera entender.

## Lectura Recomendada

La ruta base es:

```text
00_index
  -> 01_conceptual_model
  -> 02_architecture
  -> 03_data_contracts
  -> 04_fundamental_selection_article
  -> 05_correlation_selection_article
  -> 06_scoring_model
  -> 07_workflows
  -> 08_api_reference
  -> 09_validation_and_edge_cases
  -> 10_glossary
```

Si el lector solo quiere ejecutar código, puede saltar de `01` a `07`. Si quiere
modificar métricas o scoring, debe leer `03`, `04` y `06` antes de tocar
`src/selection/fundamental_scorers.py`.

## Relación Con Documentación Heredada

La documentación anterior se conserva como referencia:

| Documento Heredado | Nuevo Documento Que Lo Ordena |
|---|---|
| `architecture.md` | `02_architecture.md` |
| `api_reference.md` | `08_api_reference.md` |
| `workflows.md` | `07_workflows.md` |
| `diagrams/README.md` | `02_architecture.md` y artículos `04`/`05` |

El piloto no elimina esos documentos. Los reubica dentro de una lectura más
clara.

## Qué Leer Según La Necesidad

| Necesidad | Documento |
|---|---|
| Entender qué hace `selection` | `01_conceptual_model.md` |
| Saber cómo se conectan archivos y clases | `02_architecture.md` |
| Integrar Yahoo, un proveedor institucional o datos propios | `03_data_contracts.md` |
| Entender value/growth y selección fundamental | `04_fundamental_selection_article.md` |
| Entender correlación y diversificación | `05_correlation_selection_article.md` |
| Cambiar pesos, señales o normalización | `06_scoring_model.md` |
| Ejecutar un caso real | `07_workflows.md` |
| Consultar métodos y atributos | `08_api_reference.md` |
| Revisar límites, errores y pruebas | `09_validation_and_edge_cases.md` |

## Estado Del Piloto

Este piloto cubre `selection` como primer módulo del estándar documental. La
estructura está lista para replicarse en `optimization`, `backtesting`,
`portfolio`, `risk` y proyectos futuros.
