# Notebook Workflow: Macro/Liquidity Research To Selection Context

## Indice Del Documento

1. Nota De Estado
2. Celda 1 - Imports
3. Celda 2 - Definir Catalogo Y Datos
4. Celda 3 - Diagnostico Actual
5. Celda 4 - Forecast In-House
6. Celda 5 - Vista De Activos
7. Celda 6 - Overlay Del Cliente
8. Celda 7 - Uso En Selection
9. Salidas Esperadas

## Nota De Estado

Esta es la receta compacta de notebook. La documentacion completa del modulo
esta en [macro_liquidity/README.md](./macro_liquidity/README.md). Antes de usar
esta receta, conviene leer
[macro_liquidity/01_audit_snapshot.md](./macro_liquidity/01_audit_snapshot.md)
para distinguir providers implementados de metadata. El workflow extendido esta
en [macro_liquidity/07_workflows.md](./macro_liquidity/07_workflows.md).

Este workflow esta pensado para un notebook dentro de `research/`. Mantiene
separados tres planos:

1. Research economico: regimen actual y esperado.
2. Politica del cliente: IPS, restricciones y moneda base.
3. Traduccion estrategica: `SelectionContext` para seleccion.

## Celda 1 - Imports

```python
import pandas as pd

from src.client_policy import mexican_moderate_aggressive_growth_policy
from src.research.macro_liquidity import (
    ForecastScenario,
    LocalSeriesProvider,
    MacroLiquidityResearch,
    RegimeForecast,
    build_asset_class_view,
    default_us_macro_liquidity_catalog,
)
from src.strategy import build_selection_context
```

## Celda 2 - Definir Catalogo Y Datos

```python
catalog = default_us_macro_liquidity_catalog()

# En produccion, reemplazar por un provider API o por datos cacheados.
# Formato requerido: date, series, value.
series_frame = pd.read_csv("../data/interim/macro_liquidity_series.csv")
provider = LocalSeriesProvider(series_frame)
series_data = provider.fetch(catalog, start="2015-01-01")
```

## Celda 3 - Diagnostico Actual

```python
research = MacroLiquidityResearch(catalog, min_periods=12)
result = research.analyze_current(series_data)

result.current
result.scores.tail()
```

## Celda 4 - Forecast In-House

```python
forecast = RegimeForecast(
    scenarios=(
        ForecastScenario(
            name="base",
            macro_regime="desaceleracion_ordenada",
            liquidity_regime="restrictiva",
            probability=0.60,
            horizon_months=6,
            confidence=0.65,
            rationale="Crecimiento se modera y liquidez se vuelve menos favorable.",
        ),
        ForecastScenario(
            name="upside",
            macro_regime="expansion_desinflacionaria",
            liquidity_regime="neutral",
            probability=0.25,
            horizon_months=6,
            confidence=0.50,
            rationale="Desinflacion ordenada con condiciones financieras estables.",
        ),
        ForecastScenario(
            name="downside",
            macro_regime="estanflacion",
            liquidity_regime="estresada",
            probability=0.15,
            horizon_months=6,
            confidence=0.45,
            rationale="Inflacion persistente y deterioro de fondeo.",
        ),
    )
)

forecast.scenario_table()
```

## Celda 5 - Vista De Activos

```python
asset_view = build_asset_class_view(
    current=result.current,
    forecast=forecast,
    current_weight=0.40,
    expected_weight=0.60,
)

asset_view.to_frame()
```

## Celda 6 - Overlay Del Cliente

```python
client_policy = mexican_moderate_aggressive_growth_policy()

selection_context = build_selection_context(
    current=result.current,
    forecast=forecast,
    asset_view=asset_view,
    policy=client_policy,
)

selection_context.to_frame()
```

## Celda 7 - Uso En Selection

```python
# Ejemplo conceptual:
# - usar excluded_sectors para filtrar universo
# - usar preferred_styles y score_tilts para ajustar scoring
# - usar asset_class_overweights para decidir que universos evaluar

selection_context.score_tilts
selection_context.excluded_sectors
selection_context.asset_class_overweights
```

## Salidas Esperadas

- `result.current`: regimen macro/liquidez actual.
- `forecast.scenario_table()`: regimen esperado y escenarios alternativos.
- `asset_view.to_frame()`: postura top-down antes del cliente.
- `selection_context.to_frame()`: postura alineada al IPS y lista para
  seleccion.
