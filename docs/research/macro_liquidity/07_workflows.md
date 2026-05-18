# Workflows De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Workflow 1: Auditoria De Fuentes
4. Workflow 2: Descarga FRED
5. Workflow 3: Cache Local
6. Workflow 4: Analisis Macro/Liquidez Completo
7. Workflow 5: Handoff A Selection
8. Salidas Esperadas
9. Donde Se Rompe

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

Este documento muestra como ejecutar el flujo real con los modulos actuales. No
asume providers que no existen.

## Workflow 1: Auditoria De Fuentes

Objetivo: saber que fuentes tienen credenciales disponibles.

```python
from src.research.macro_liquidity import ProviderConfig

config = ProviderConfig()
availability = config.availability_report()
availability[["provider", "requires_key", "env_var", "token_configured", "available"]]
```

Interpretacion importante:

- `available=True` significa que no falta credencial segun metadata.
- No significa que exista provider operativo.
- Hoy solo `FredApiProvider` y `LocalSeriesProvider` son providers operativos.

## Workflow 2: Descarga FRED

Objetivo: descargar las series FRED del catalogo.

Precondicion:

```bash
export FRED_API_KEY="tu_key"
```

Codigo:

```python
from src.research.macro_liquidity import (
    FredApiProvider,
    ProviderConfig,
    default_us_macro_liquidity_catalog,
)

catalog = default_us_macro_liquidity_catalog()
fred_specs = tuple(spec for spec in catalog if spec.provider.upper() == "FRED")

provider = FredApiProvider(config=ProviderConfig(), timeout=30)
series_data = provider.fetch(fred_specs, start="2015-01-01")

series_data.head()
```

Salida: `DataFrame` con `date`, `series`, `value`.

## Workflow 3: Cache Local

Objetivo: correr research sin depender de red o credenciales.

```python
import pandas as pd

from src.research.macro_liquidity import (
    LocalSeriesProvider,
    default_us_macro_liquidity_catalog,
)

catalog = default_us_macro_liquidity_catalog()
series_frame = pd.read_csv("../data/interim/macro_liquidity_series.csv")

provider = LocalSeriesProvider(series_frame)
series_data = provider.fetch(catalog, start="2015-01-01")
```

El CSV debe tener columnas `date`, `series`, `value`.

## Workflow 4: Analisis Macro/Liquidez Completo

Objetivo: transformar datos en regimen actual, forecast, vista de activos y
contexto para selection.

```python
from src.client_policy import mexican_moderate_aggressive_growth_policy
from src.research.macro_liquidity import (
    ForecastScenario,
    MacroLiquidityResearch,
    RegimeForecast,
    build_asset_class_view,
)
from src.strategy import build_selection_context

research = MacroLiquidityResearch(catalog, min_periods=12)
result = research.analyze_current(series_data)

forecast = RegimeForecast(
    scenarios=(
        ForecastScenario(
            name="base",
            macro_regime="desaceleracion_ordenada",
            liquidity_regime="restrictiva",
            probability=0.60,
            horizon_months=6,
            confidence=0.65,
            rationale="House view: crecimiento se modera y liquidez menos favorable.",
        ),
        ForecastScenario(
            name="upside",
            macro_regime="expansion_desinflacionaria",
            liquidity_regime="neutral",
            probability=0.25,
            horizon_months=6,
            confidence=0.50,
        ),
        ForecastScenario(
            name="downside",
            macro_regime="estanflacion",
            liquidity_regime="estresada",
            probability=0.15,
            horizon_months=6,
            confidence=0.45,
        ),
    )
)

asset_view = build_asset_class_view(
    current=result.current,
    forecast=forecast,
    current_weight=0.40,
    expected_weight=0.60,
)

policy = mexican_moderate_aggressive_growth_policy()

selection_context = build_selection_context(
    current=result.current,
    forecast=forecast,
    asset_view=asset_view,
    policy=policy,
)
```

## Workflow 5: Handoff A Selection

Objetivo: usar `SelectionContext` como contrato downstream.

```python
selection_context.as_dict()
selection_context.to_frame()
selection_context.score_tilts
selection_context.excluded_sectors
selection_context.eligible_assets
```

Uso esperado en selection:

- Filtrar por `excluded_sectors`.
- Respetar `eligible_assets`.
- Ajustar scoring con `score_tilts`.
- Revisar `risk_controls`.
- Mantener `rationale` para trazabilidad.

## Salidas Esperadas

| Etapa | Objeto | Uso |
|---|---|---|
| Datos | `series_data` | Input normalizado del research. |
| Diagnostico | `MacroLiquidityResult` | Incluye indicadores, scores y snapshot. |
| Snapshot | `CurrentRegimeSnapshot` | Regimen actual. |
| Forecast | `RegimeForecast` | Regimen esperado in-house. |
| Vista de activos | `AssetClassView` | Tilts top-down antes del cliente. |
| Handoff | `SelectionContext` | Contrato para selection. |

## Donde Se Rompe

- Falta `FRED_API_KEY`: `MissingCredentialError`.
- FRED responde error: `requests.HTTPError`.
- CSV local no tiene columnas esperadas: `ValueError`.
- No hay datos scoreables: `ValueError` por scores vacios.
- Forecast sin escenarios: `ValueError`.
- Datos con frecuencias mezcladas: el flujo corre, pero el snapshot puede ser
  metodologicamente fragil.
