# Selection Module

El módulo `src.selection` concentra la lógica de selección de activos antes de
pasar a optimización, backtesting o análisis de riesgo. Su objetivo es responder
una pregunta previa a la construcción del portafolio:

> De un universo inicial de compañías, ¿cuáles merecen entrar al conjunto de
> candidatos que después se va a optimizar o analizar?

Actualmente el módulo cubre dos familias de selección:

- Selección por correlación, útil para construir universos diversificados.
- Selección fundamental, útil para rankear compañías con criterios de `value`
  investing o `growth` investing usando información de Yahoo Finance.
- Análisis fundamental temporal, útil para revisar la evolución trimestral o
  anual de métricas y scores.

## Estructura

```text
src/selection/
├── __init__.py
├── correlation_selector.py
├── fundamentals.py
├── fundamental_metrics.py
├── fundamental_scorers.py
└── fundamental_selector.py
```

## Documentación

- [architecture.md](architecture.md): relación entre módulos, clases y flujo de datos.
- [api_reference.md](api_reference.md): referencia de clases, atributos, métodos y funciones.
- [workflows.md](workflows.md): casos de uso y flujos de trabajo recomendados.

## Flujo Orgánico

El uso más natural del módulo empieza con un universo amplio de tickers y termina
con una lista compacta de candidatos.

```python
from src.selection import FundamentalSelector

selector = FundamentalSelector(strategy="value")
ranking = selector.rank(["AAPL", "MSFT", "NVDA", "KO"])
selected = selector.select_top(ranking, top_k=3)
```

El resultado `selected` es un `DataFrame` con las compañías mejor rankeadas, sus
métricas fundamentales y su `fundamental_score`. Ese resultado puede alimentar
los módulos posteriores del proyecto:

```python
selected_tickers = selected["ticker"].tolist()
```

A partir de ahí, `selected_tickers` puede usarse en investigación, optimización,
backtesting o monitoreo de riesgo.

Para revisar el score a través del tiempo:

```python
score_history = selector.rank_over_time(
    ["AAPL", "MSFT", "NVDA", "KO"],
    frequency="quarterly",
    trailing_periods=4,
)
```

Para explicar la selección final:

```python
report = selector.selection_report(ranking, top_k=3)
```

`report` resume los tickers seleccionados, las métricas utilizadas, sus pesos y
los componentes de score.

## Principios De Diseño

El diseño del módulo sigue tres principios:

1. Separar adquisición de datos, cálculo de métricas y ranking.
2. Mantener funciones puras para que las métricas y scores sean testeables.
3. Exponer una API simple para notebooks e investigación exploratoria.

La separación es intencional. Yahoo Finance puede cambiar, devolver campos
faltantes o comportarse de forma irregular. Por eso `fundamentals.py` aísla la
descarga, mientras que `fundamental_metrics.py` y `fundamental_scorers.py`
trabajan sobre estructuras de `pandas`.

El proveedor de Yahoo Finance mantiene una caché por ticker, rango de fechas y
campo de precio. Esto evita repetir consultas cuando se calculan métricas,
historiales y reportes sobre el mismo universo.

## Convenciones

La documentación usa una estructura similar a NumPy y pandas:

- Descripción breve de cada clase o función.
- Parámetros.
- Atributos.
- Métodos.
- Retornos.
- Notas.
- Ejemplos.

Los nombres públicos usan `snake_case` para funciones y métodos, y `CamelCase`
para clases, siguiendo PEP 8.
