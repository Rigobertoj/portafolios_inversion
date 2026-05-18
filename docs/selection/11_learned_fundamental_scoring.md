# Selection: Scoring Fundamental Aprendido

## Propósito

Este documento explica la extensión aprendida de selección fundamental. La idea
es separar research y operación:

```text
panel histórico -> modelo XGBoost -> impacto de señales -> pesos aprendidos
-> FundamentalScoreConfig -> ranking fundamental normal
```

El modelo aprendido no reemplaza `score_fundamentals`. Produce evidencia para
construir configuraciones compatibles con `MetricSignalSpec`.

## Dependencia Opcional

El flujo value/growth normal no necesita XGBoost. Para entrenar
`XGBoostFundamentalModel` con el estimador default, instalar el extra de
machine learning:

```bash
python -m pip install -e ".[ml]"
```

También se puede instalar directamente:

```bash
python -m pip install xgboost
```

## Flujo

```python
from src.selection import (
    FundamentalLearningPanelBuilder,
    LearnedFundamentalSelector,
    LearnedScoreConfigFactory,
    XGBoostFundamentalModel,
)

panel = FundamentalLearningPanelBuilder(
    frequency="quarterly",
    trailing_periods=20,
    horizon_months=12,
    reporting_lag_days=60,
).build(["AAPL", "MSFT", "NVDA", "KO"])

model = XGBoostFundamentalModel(
    target="forward_total_return_12m",
    group_by="sector",
    min_group_size=30,
)
model.fit(panel)

configs = LearnedScoreConfigFactory(max_features=15).from_impact_report(
    model.impact_report_
)

selector = LearnedFundamentalSelector(
    score_configs_by_group=configs,
    group_by="sector",
)
ranking = selector.rank(["AAPL", "MSFT", "NVDA", "KO"])
selected = selector.select_top(ranking, top_k=3)
```

## Componentes

| Archivo | Responsabilidad |
|---|---|
| `fundamental_targets.py` | Construye `forward_price_return`, `forward_dividend_return` y `forward_total_return`. |
| `fundamental_panel.py` | Construye el panel histórico con señales `metric__signal`. |
| `xgboost_fundamental_model.py` | Ajusta modelos por grupo y produce `impact_report_`. |
| `learned_fundamental_scorers.py` | Convierte impactos en `FundamentalScoreConfig`. |
| `learned_fundamental_selector.py` | Rankea empresas usando configs aprendidas por sector o industria. |

## Targets Forward

Un target forward mide lo que ocurre después de que la información fundamental
se considera disponible:

```text
available_at = period + reporting_lag_days
target_end_at = available_at + horizon_months

forward_price_return = exit_price / entry_price - 1
forward_dividend_return = dividends_paid / entry_price
forward_total_return = price_return + dividend_return
```

El rezago reduce riesgo de look-ahead porque el modelo no usa estados
financieros el mismo día del cierre contable.

## Conversión A Pesos

`XGBoostFundamentalModel` produce columnas como:

- `importance`
- `higher_is_better`
- `stability`
- `oos_score`
- `coverage`
- `adjusted_impact`

La fábrica usa `adjusted_impact` como fuente de pesos:

```text
weight_j = adjusted_impact_j / sum(adjusted_impact)
```

Si todos los impactos ajustados son cero, puede caer a `importance` para no
perder el ranking exploratorio. Esa caída debe revisarse antes de usar el score
en una decisión real.

## Sesgos Controlados Por Diseño

- Look-ahead: targets empiezan después de `reporting_lag_days`.
- Overfitting: el modelo calcula diagnóstico walk-forward con rank IC.
- Sesgo sectorial: puede entrenar por `sector`, `industry` o usar fallback
  global.
- Datos faltantes: cada feature exige cobertura mínima y el score conserva
  `score_coverage`.
- Dependencia opcional: `xgboost` solo se requiere al ajustar el modelo
  default. Los selectores manuales siguen funcionando sin instalarlo.

## Límites

Yahoo Finance no es una fuente point-in-time institucional. Para research
defendible se necesita controlar survivorship, fechas reales de disponibilidad,
restatements y delistings. Esta extensión deja la arquitectura lista para eso,
pero no convierte automáticamente datos gratuitos en una base institucional.
