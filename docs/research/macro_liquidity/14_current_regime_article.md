# Articulo Complementario: Diagnostico De Regimen Actual

## Indice Del Documento

1. Mapa De La Serie
2. Problema Que Resuelve
3. Fundamento Conceptual
4. Fuentes Academicas O Profesionales
5. Traduccion Al Codigo
6. Clases, Metodos Y Atributos Implicados
7. Ejemplo Interpretado
8. Supuestos Y Limites
9. Errores Comunes
10. Relacion Con Workflows
11. Referencias

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
| 14 | `14_current_regime_article.md` | Articulo complementario. |

## Problema Que Resuelve

El diagnostico de regimen actual resume muchas series economicas en una lectura
compacta:

```text
macro_regime + liquidity_regime + confidence
```

Esta lectura permite que el research no dependa de una sola variable como PIB,
inflacion o tasas. La salida es `CurrentRegimeSnapshot`, que despues alimenta
`AssetClassView` y `SelectionContext`.

## Fundamento Conceptual

La logica del diagnostico toma dos ideas:

- Un ciclo economico se observa en varias dimensiones de actividad agregada,
  empleo, consumo, produccion, ingreso e inflacion.
- La liquidez y las condiciones financieras pueden moverse de forma distinta al
  crecimiento tradicional y afectar la asimetria riesgo-retorno de los activos.

Por eso el catalogo separa `engine="macro"` de `engine="liquidity"` y agrupa
series por bloques como `growth`, `inflation`, `labor`, `policy`, `credit` y
`financial_conditions`.

## Fuentes Academicas O Profesionales

El enfoque de multiples indicadores es consistente con el criterio de NBER para
ciclos economicos, que no depende de una sola regla mecanica. Para liquidez, el
NFCI de Chicago Fed es una referencia practica porque resume condiciones de
dinero, deuda, equity, banca tradicional y banca sombra en una escala
normalizada.

El modulo no replica exactamente NBER ni NFCI; toma su principio profesional:
usar varias variables, normalizarlas y mantener separadas actividad economica y
condiciones financieras.

## Traduccion Al Codigo

El diagnostico se implementa en tres pasos:

1. `build_indicator_panel(series_frame, specs, min_periods)`:
   normaliza datos, aplica transformaciones y calcula `signed_zscore`.
2. `build_score_frame(indicators)`:
   agrega scores por bloque y por engine.
3. `MacroLiquidityResearch.analyze_current(series_frame)`:
   clasifica el ultimo score disponible en `CurrentRegimeSnapshot`.

La clasificacion de liquidez usa `classify_liquidity_regime(score)`:

| Score | Regimen |
|---:|---|
| `>= 1.0` | `expansiva` |
| `>= 0.5` | `moderadamente_expansiva` |
| `> -0.5` | `neutral` |
| `> -1.0` | `restrictiva` |
| `<= -1.0` | `estresada` |

La clasificacion macro usa `classify_macro_regime(row)`, que combina crecimiento,
inflacion, empleo y politica.

## Clases, Metodos Y Atributos Implicados

| Objeto | Archivo | Papel |
|---|---|---|
| `EconomicSeriesSpec` | `catalog.py` | Define que significa cada serie y como debe transformarse. |
| `MacroLiquidityResearch` | `regimes.py` | Orquesta el diagnostico actual. |
| `MacroLiquidityResult` | `regimes.py` | Agrupa indicadores, scores y snapshot. |
| `CurrentRegimeSnapshot` | `regimes.py` | Salida de regimen actual. |
| `build_indicator_panel` | `transforms.py` | Estado intermedio auditable. |
| `build_score_frame` | `transforms.py` | Agregacion a scores por bloque y engine. |

La relacion completa esta representada en
[macro_liquidity_class_architecture.drawio](./diagrams/macro_liquidity_class_architecture.drawio).

## Ejemplo Interpretado

En `test/research/test_macro_liquidity_research.py`, la muestra sintetica usa:

- Crecimiento en `real_gdp`.
- Inflacion descendente en `cpi`.
- Empleo fuerte via `unemployment_rate`.
- Liquidez mejorando por `reserve_balances`, `tga` descendente y `hy_oas`
  comprimiendose.

Con `min_periods=3`, el test espera:

```text
macro_regime = expansion_desinflacionaria
liquidity_regime = expansiva
```

La interpretacion es que el panel sintetico muestra crecimiento positivo,
inflacion menos restrictiva y liquidez favorable. El resultado no es una
recomendacion de compra; es una condicion inicial para el flujo posterior.

## Supuestos Y Limites

- Los z-scores son historicos y expansivos; una ventana corta puede producir
  lecturas inestables.
- `higher_is_better` traduce cada serie a una direccion economica; si esa
  direccion esta mal definida, el score queda invertido.
- Las frecuencias mixtas se ordenan por fecha, pero no se hace nowcasting ni
  alineacion sofisticada de publicaciones.
- La confianza del snapshot mide cobertura, no precision predictiva.
- Los umbrales son interpretables, no calibrados todavia por backtesting.

## Errores Comunes

- Leer `macro_score` como pronostico de retorno esperado.
- Ignorar `macro_coverage` y `liquidity_coverage`.
- Mezclar series con unidades y frecuencias sin validar fechas.
- Interpretar `unknown` como neutral.
- Usar el resultado sin revisar si faltan series requeridas.

## Relacion Con Workflows

El flujo de notebook recomendado esta en `07_workflows.md`. El paso critico es
revisar `result.scores.tail()` y `result.current` antes de definir el forecast
in-house.

## Referencias

- NBER, Business Cycle Dating. URL:
  https://www.nber.org/research/business-cycle-dating. Consultado:
  2026-05-08. Referencia para ciclos economicos medidos con multiples
  indicadores.
- NBER, Business Cycle Dating Procedure FAQ. URL:
  https://www.nber.org/research/business-cycle-dating/business-cycle-dating-procedure-frequently-asked-questions.
  Consultado: 2026-05-08. Referencia para profundidad, difusion y duracion.
- Federal Reserve Bank of Chicago, National Financial Conditions Index. URL:
  https://www.chicagofed.org/research/data/nfci/about?trigger=true. Consultado:
  2026-05-08. Referencia para condiciones financieras normalizadas.
- Federal Reserve Bank of St. Louis, FRED API Overview. URL:
  https://fred.stlouisfed.org/docs/api/fred/overview.html. Consultado:
  2026-05-08. Referencia tecnica para descarga programatica de series.
