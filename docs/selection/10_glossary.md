# Selection: Glosario

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Términos Del Módulo
4. Términos De Scoring
5. Términos Financieros
6. Términos De Correlación

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

Este glosario fija el vocabulario usado en la documentación de `selection`.

## Términos Del Módulo

| Término | Definición |
|---|---|
| Universo | Conjunto inicial de tickers evaluados. |
| Candidato | Activo que pasa filtros o ranking y puede alimentar módulos posteriores. |
| Selección | Proceso de reducir un universo a candidatos con evidencia. |
| Handoff | Contrato que recibe el siguiente módulo, normalmente `selected_tickers`. |
| Evidencia | Métricas, scores, componentes, matrices o reportes que justifican la selección. |

## Términos De Scoring

| Término | Definición |
|---|---|
| Métrica | Columna financiera base, como `roe` o `trailing_pe`. |
| Señal | Transformación de una métrica, como nivel o cambio histórico. |
| Componente | Señal normalizada que participa en el score. |
| Peso | Importancia relativa de un componente. |
| Dirección | Indica si valores altos o bajos son preferibles. |
| Normalizador | Método que lleva valores heterogéneos a escala común. |
| `fundamental_score` | Score compuesto final en escala 0-100. |
| `score_coverage` | Proporción del peso total con datos disponibles. |

## Términos Financieros

| Término | Definición |
|---|---|
| Value | Tesis que favorece valuación atractiva con calidad financiera. |
| Growth | Tesis que favorece expansión de ventas, utilidades o cash flow. |
| Múltiplo | Relación entre precio o valor de empresa y una variable fundamental. |
| Rentabilidad | Capacidad de generar utilidad o retorno sobre capital. |
| Liquidez | Capacidad de cubrir obligaciones de corto plazo. |
| Leverage | Uso de deuda en la estructura financiera. |
| Cash flow | Flujo de efectivo generado o disponible. |

## Términos De Correlación

| Término | Definición |
|---|---|
| Retorno | Cambio porcentual de precio entre periodos. |
| Correlación | Medida de co-movimiento lineal entre retornos. |
| `corr_score` | Correlación promedio de un ticker contra sus pares. |
| `use_absolute_corr` | Opción que usa magnitud absoluta de correlación. |
| `min_coverage` | Cobertura mínima de precios para conservar un activo. |
| Correlación fuera de diagonal | Correlaciones entre activos distintos dentro de una matriz. |

