# Selection: Artículo De Selección Por Correlación

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Problema Que Resuelve
4. Fundamento Conceptual
5. Fuentes Académicas O Profesionales
6. Traducción Al Código
7. Interpretación De `corr_score`
8. Ejemplo Interpretado
9. Supuestos Y Límites
10. Errores Comunes
11. Referencias

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

Este artículo explica la ruta de correlación de `selection`. Su objetivo es
mostrar cómo una intuición de diversificación se convierte en un ranking
operativo de activos.

## Problema Que Resuelve

La selección por correlación responde:

```text
¿Qué activos agregan menos redundancia estadística al universo o al portafolio?
```

La ruta no intenta identificar empresas subvaluadas ni calcular pesos óptimos.
Busca candidatos que, por comportamiento histórico de retornos, sean menos
similares a sus pares.

## Fundamento Conceptual

En teoría de portafolios, la correlación entre activos es central para entender
riesgo agregado. CFA Institute explica que, al aumentar el número de activos,
la correlación entre riesgos se vuelve un determinante importante del riesgo de
portafolio. Markowitz formalizó la selección de portafolios como una decisión
que combina retorno esperado, varianza y covarianza.

`selection` toma una parte de esa intuición: antes de optimizar pesos, puede
filtrar candidatos con menor correlación promedio dentro de grupos.

## Fuentes Académicas O Profesionales

- Harry Markowitz, *Portfolio Selection*, Journal of Finance, 1952. Base de la
  teoría moderna de portafolios y del rol de covarianzas.
- CFA Institute, *Portfolio Risk and Return: Part I*. Referencia profesional
  sobre riesgo, retorno, correlación, diversificación y frontera eficiente.

## Traducción Al Código

La clase principal es `CorrelationPortfolioSelector`.

```text
rank_within_groups(grouped_tickers)
  -> descarga precios
  -> aplica filtro de cobertura
  -> calcula retornos
  -> calcula matriz de correlación
  -> _compute_corr_scores
  -> ranking_by_group
```

Luego:

```text
build_multigroup_portfolio
  -> toma candidatos por grupo
  -> calcula correlación entre candidatos
  -> rankea por menor corr_score
  -> devuelve final_tickers
```

## Interpretación De `corr_score`

`corr_score` es la correlación promedio de un ticker contra sus pares. Si
`use_absolute_corr=True`, se usa la magnitud absoluta de la correlación.

Interpretación:

| Valor | Lectura |
|---:|---|
| Bajo | Activo menos parecido a sus pares. |
| Alto | Activo más redundante dentro del grupo. |
| Negativo, si `use_absolute_corr=False` | Relación inversa promedio. |

El score no indica mejor empresa, mayor retorno esperado ni menor riesgo
individual. Indica menor similitud estadística histórica.

## Ejemplo Interpretado

```python
from src.selection import CorrelationPortfolioSelector

selector = CorrelationPortfolioSelector(
    start_date="2020-01-01",
    end_date="2025-01-01",
    use_absolute_corr=True,
)

result = selector.run_pipeline(
    grouped_tickers={
        "tech": ["AAPL", "MSFT", "NVDA"],
        "defensive": ["KO", "PG", "WMT"],
    },
    top_k_in_group=1,
)
```

Lectura del resultado:

- `group_ranking` explica qué activo fue menos correlacionado por grupo.
- `candidate_tickers` consolida los ganadores por grupo.
- `candidate_corr_matrix` muestra la relación entre candidatos.
- `final_tickers` es la lista candidata para la siguiente etapa.
- `final_mean_offdiag_corr` resume la correlación promedio del conjunto final.

## Supuestos Y Límites

- La correlación se calcula sobre datos históricos.
- La correlación puede cambiar por régimen de mercado.
- El filtro `min_coverage` puede excluir activos con datos incompletos.
- Menor correlación no garantiza mejor retorno.
- Menor correlación no sustituye optimización de pesos.
- Si el grupo tiene menos de dos activos útiles, no se puede rankear.

## Errores Comunes

- Interpretar baja correlación como recomendación de compra.
- Ignorar que `use_absolute_corr=True` trata correlaciones negativas fuertes
  como alta relación estadística.
- Mezclar grupos sin justificar su definición.
- Usar ventanas históricas demasiado cortas.
- Pasar a optimización sin revisar la matriz de correlación final.

## Referencias

- Markowitz, Harry. "Portfolio Selection." *Journal of Finance*, 7(1), 77-91,
  1952. Ficha bibliográfica consultada: 2026-05-03.
  https://ideas.repec.org/a/bla/jfinan/v7y1952i1p77-91.html
- CFA Institute, *Portfolio Risk and Return: Part I*. Consultado: 2026-05-03.
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/portfolio-risk-return-part-1

