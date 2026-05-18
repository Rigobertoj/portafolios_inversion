# Ejemplos De Uso De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Ejemplo Minimo Con Datos Locales
4. Ejemplo Con FRED
5. Ejemplo De Forecast
6. Ejemplo De Handoff

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

Ejemplos cortos para copiar en notebooks. Para explicacion completa, leer
`07_workflows.md`.

## Ejemplo Minimo Con Datos Locales

```python
import pandas as pd

from src.research.macro_liquidity import (
    LocalSeriesProvider,
    MacroLiquidityResearch,
    default_us_macro_liquidity_catalog,
)

catalog = default_us_macro_liquidity_catalog()
series_frame = pd.read_csv("../data/interim/macro_liquidity_series.csv")
series_data = LocalSeriesProvider(series_frame).fetch(catalog, start="2015-01-01")

result = MacroLiquidityResearch(catalog, min_periods=12).analyze_current(series_data)
result.current
```

## Ejemplo Con FRED

```python
from src.research.macro_liquidity import (
    FredApiProvider,
    ProviderConfig,
    default_us_macro_liquidity_catalog,
)

catalog = default_us_macro_liquidity_catalog()
fred_specs = tuple(spec for spec in catalog if spec.provider.upper() == "FRED")

series_data = FredApiProvider(ProviderConfig()).fetch(
    fred_specs,
    start="2015-01-01",
)
```

Requiere `FRED_API_KEY` en el ambiente.

## Ejemplo De Forecast

```python
from src.research.macro_liquidity import ForecastScenario, RegimeForecast

forecast = RegimeForecast(
    scenarios=(
        ForecastScenario(
            name="base",
            macro_regime="desaceleracion_ordenada",
            liquidity_regime="restrictiva",
            probability=0.60,
            horizon_months=6,
            confidence=0.65,
        ),
    )
)

forecast.scenario_table()
```

## Ejemplo De Handoff

```python
from src.client_policy import mexican_moderate_aggressive_growth_policy
from src.research.macro_liquidity import build_asset_class_view
from src.strategy import build_selection_context

asset_view = build_asset_class_view(result.current, forecast)
policy = mexican_moderate_aggressive_growth_policy()

context = build_selection_context(result.current, forecast, asset_view, policy)
context.to_frame()
```
