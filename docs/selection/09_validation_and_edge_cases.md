# Selection: Validación Y Edge Cases

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Validaciones De Entrada
4. Edge Cases Fundamentales
5. Edge Cases De Scoring
6. Edge Cases De Correlación
7. Tests Relevantes
8. Checklist Antes De Usar Salidas

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

Este documento lista validaciones, errores frecuentes y riesgos de uso. Debe
leerse antes de depender de `selected_tickers` en optimización o backtesting.

## Validaciones De Entrada

| Caso | Resultado Esperado |
|---|---|
| Tickers vacíos | `ValueError` en normalización fundamental. |
| `top_k < 1` | `ValueError`. |
| Estrategia desconocida | `ValueError` en `config_for_strategy`. |
| Grupo con menos de dos tickers útiles | Se omite o falla si no queda ranking. |
| `max_new_assets < 1` | `ValueError`. |

## Edge Cases Fundamentales

- Yahoo Finance puede devolver estados financieros vacíos.
- Algunas métricas no existen para todas las compañías.
- El historial puede no tener periodos suficientes para `yoy_change`.
- Un score alto con baja cobertura no debe interpretarse como evidencia fuerte.
- Empresas de sectores distintos pueden no ser comparables bajo la misma métrica.

## Edge Cases De Scoring

- Si ningún componente tiene datos, el score queda en 0.
- Si una métrica tiene un solo valor válido, el ranking puede ser poco
  informativo.
- Los percentiles cambian cuando cambia el universo.
- `winsorize_quantiles` reduce impacto de outliers, pero no corrige datos malos.
- `missing_component_score=0.0` penaliza falta de datos en componentes
  existentes.

## Edge Cases De Correlación

- Una ventana histórica corta puede generar correlaciones inestables.
- `use_absolute_corr=True` penaliza correlaciones negativas fuertes.
- `min_coverage` puede dejar grupos sin suficientes activos.
- Cambios de régimen pueden invalidar correlaciones históricas.
- Baja correlación no implica baja volatilidad.

## Tests Relevantes

La carpeta `test/selection/` contiene pruebas para:

- Métricas fundamentales.
- Scorers fundamentales.
- Selector fundamental.
- Historial fundamental.
- Proveedor fundamental.

Antes de modificar scoring, revisar especialmente:

```text
test/selection/test_fundamental_scorers.py
test/selection/test_fundamental_selector.py
test/selection/test_fundamental_metrics.py
```

## Checklist Antes De Usar Salidas

- [ ] Revisar `score_coverage`.
- [ ] Confirmar que el universo comparado sea coherente.
- [ ] Revisar métricas y componentes principales del reporte.
- [ ] Verificar que no se mezclen proveedores sin normalización.
- [ ] En correlación, revisar `candidate_corr_matrix`.
- [ ] Confirmar que `selected_tickers` no se interprete como pesos.

