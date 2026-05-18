# Confidence, Forecast Y Scenario Policy En `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Audit Snapshot
4. Modelo Mental
5. Donde Vive Cada Pieza
6. `ConfidencePolicy`
7. `ForecastPolicy`
8. Auto SARIMA Frente A AR(1)
9. `ScenarioPolicy`
10. Lectura Operativa En Notebook
11. Riesgos Y Limites
12. Checklist De Revision

## Proposito

Este documento explica como el modulo `macro_liquidity` convierte evidencia
macroeconomica y de liquidez en tres salidas intermedias:

- confianza del diagnostico actual;
- pronosticos de scores macro/liquidez;
- escenarios base, upside y downside.

El objetivo es evitar que estas piezas parezcan una caja negra. Cada calculo se
presenta como una decision metodologica auditable: que datos recibe, que formula
aplica, que significa el resultado, como influye en el flujo posterior y donde
puede intervenir una persona.

## Mapa De La Serie

| Documento | Rol |
|---|---|
| `01_audit_snapshot.md` | Estado real del modulo. |
| `02_conceptual_model.md` | Modelo economico general. |
| `04_architecture.md` | Relacion entre componentes. |
| `07_workflows.md` | Uso operativo en notebook. |
| `09_classes_and_methods.md` | Referencia de clases y metodos. |
| `17_confidence_forecast_scenario_policy.md` | Explicacion analitica de confianza, forecast y escenarios. |

## Audit Snapshot

| Elemento | Estado | Evidencia | Riesgo / Siguiente Accion |
|---|---|---|---|
| `ConfidencePolicy` | Implementado | `src/research/macro_liquidity/policy.py` y `nowcast.py` | Es un indice de calidad del diagnostico, no una probabilidad de acierto. |
| `MacroLiquidityNowcaster` | Implementado | `nowcast.py` | La coherencia narrativa se mide solo entre `macro_score` y `liquidity_score`. |
| Forecast AR(1) | Implementado | `forecasting.py` | Es baseline; no selecciona automaticamente otros modelos. |
| Diagnosticos de forecast | Implementado | `forecasting.py` | Incluye RMSE, MAE, AIC, BIC, ACF y volatilidad EWMA de residuos. |
| `ScenarioPolicy` | Implementado | `scenarios.py` | Las probabilidades son mecanicas y deben revisarse en research. |
| Overrides humanos | Implementado | `ScenarioOverride` | Deben llevar rationale y aprobador. |
| Auto SARIMA | Planeado | No hay clase operativa aun | Requiere `ModelSelector`, backtest rolling y comparacion contra AR(1). |

## Modelo Mental

El flujo relevante es:

```text
series_data
  -> FeatureBuilder
  -> MacroLiquidityNowcaster
  -> MacroLiquidityForecaster
  -> RegimeClassifier
  -> ScenarioEngine
  -> AssetClassView
  -> SelectionContext
```

Las tres politicas principales no hacen lo mismo:

| Politica | Pregunta Que Responde | Salida |
|---|---|---|
| `ConfidencePolicy` | Que tan defendible es el diagnostico actual? | `confidence` entre 0 y 1. |
| `ForecastPolicy` | Como se proyectan los scores hacia 3, 12 y 6 meses? | tabla de forecast y diagnosticos. |
| `ScenarioPolicy` | Como se convierten forecasts en escenarios accionables? | `RegimeForecast` con base/upside/downside. |

La distincion critica es esta:

```text
regime = que cree el sistema que esta pasando
confidence = que tan solida es la evidencia para sostener ese diagnostico
forecast = hacia donde se espera que se muevan los scores
scenario = como se traduce esa trayectoria a casos de decision
```

## Donde Vive Cada Pieza

| Pieza | Archivo | Responsabilidad |
|---|---|---|
| `ConfidencePolicy` | `policy.py` | Pesos del indice de confianza. |
| `MacroLiquidityNowcaster` | `nowcast.py` | Calcula scores actuales, contribuidores y confianza. |
| `ForecastPolicy` | `policy.py` | Horizontes, benchmark y parametros de intervalos. |
| `MacroLiquidityForecaster` | `forecasting.py` | Ajusta AR(1), genera forecast y diagnosticos. |
| `ScenarioPolicy` | `policy.py` | Reglas de probabilidad base/upside/downside. |
| `ScenarioEngine` | `scenarios.py` | Construye escenarios y aplica overrides. |
| `MacroLiquidityWorkflow` | `pipeline.py` | Orquesta todo y escribe auditoria. |

## `ConfidencePolicy`

### Que Mide

`ConfidencePolicy` mide la calidad del diagnostico actual. No mide retorno
esperado, no mide probabilidad de acierto y no dice si el regimen es bueno o
malo.

Un diagnostico puede decir:

```text
macro_regime = desaceleracion_ordenada
liquidity_regime = restrictiva
```

La confianza responde otra pregunta:

```text
Ese diagnostico esta sostenido por suficientes datos, senales alineadas,
estabilidad reciente, datos frescos y coherencia macro-liquidez?
```

### Datos Que Recibe

El calculo usa tres objetos que ya vienen del flujo anterior:

| Entrada | Productor | Campos Relevantes |
|---|---|---|
| `features` | `FeatureBuilder` | `signed_zscore`, `weight`, `publication_lag_days`. |
| `scores` | `build_score_frame` | `macro_score`, `liquidity_score`, `macro_coverage`, `liquidity_coverage`. |
| `block_scores` | `MacroLiquidityNowcaster._block_scores` | score, coverage y diffusion por bloque. |

El campo mas importante de `features` es `signed_zscore`.

```text
signed_zscore > 0  -> senal favorable para su bloque
signed_zscore < 0  -> senal desfavorable para su bloque
```

El signo ya incorpora la direccion economica de cada variable. Por ejemplo:

| Variable | Si Sube | Direccion |
|---|---|---|
| payrolls | favorable | `higher_is_better=True` |
| desempleo | desfavorable | `higher_is_better=False` |
| spreads de credito | desfavorable | `higher_is_better=False` |
| reservas bancarias | favorable | `higher_is_better=True` |

### Componentes

La confianza actual se calcula como:

```text
confidence =
  coverage * coverage_weight
  + agreement * agreement_weight
  + stability * stability_weight
  + freshness * freshness_weight
  + narrative_coherence * narrative_weight
```

Con los parametros por defecto:

```text
coverage_weight = 0.35
agreement_weight = 0.25
stability_weight = 0.20
freshness_weight = 0.10
narrative_weight = 0.10
contradiction_threshold = 0.50
```

### `coverage`

`coverage` mide cuantas senales validas existen para calcular el diagnostico.

Primero el sistema calcula cobertura macro y cobertura de liquidez:

```text
macro_coverage = indicadores macro con signed_zscore valido / indicadores macro totales
liquidity_coverage = indicadores liquidez con signed_zscore valido / indicadores liquidez totales
```

Despues:

```text
coverage = promedio(macro_coverage, liquidity_coverage)
```

Ejemplo:

```text
macro_coverage = 15 / 20 = 0.75
liquidity_coverage = 9 / 10 = 0.90
coverage = (0.75 + 0.90) / 2 = 0.825
```

Interpretacion:

```text
coverage alto -> hay suficiente evidencia disponible
coverage bajo -> el diagnostico depende de pocos datos
```

No significa que la economia este bien. Solo significa que hay informacion
suficiente.

### `agreement`

`agreement` mide si las senales disponibles apuntan en una direccion comun.

El sistema calcula:

```text
positive_share = porcentaje de indicadores con signed_zscore > 0
agreement = 2 * abs(positive_share - 0.5)
```

Ejemplos:

| Positive Share | Agreement | Lectura |
|---:|---:|---|
| 1.00 | 1.00 | Todas las senales son favorables. |
| 0.50 | 0.00 | Senales divididas. |
| 0.20 | 0.60 | Mayor parte de senales desfavorables. |

Punto importante:

```text
agreement alto no significa senales positivas.
agreement alto significa senales alineadas.
```

Un deterioro amplio y consistente puede producir `agreement` alto.

### `stability`

`stability` mide si los scores agregados han sido estables recientemente.

El sistema toma los ultimos seis meses de:

```text
macro_score
liquidity_score
```

Calcula su volatilidad promedio y transforma esa volatilidad a un indicador
entre 0 y 1:

```text
stability = 1 / (1 + recent_volatility)
```

Ejemplos:

| Volatilidad Reciente | Stability |
|---:|---:|
| 0.20 | 0.83 |
| 0.50 | 0.67 |
| 1.00 | 0.50 |

Interpretacion:

```text
stability alta -> el diagnostico no cambia violentamente mes a mes
stability baja -> hay transicion, ruido o inestabilidad de regimen
```

### `freshness`

`freshness` mide que tan recientes son los datos usados.

El sistema usa `publication_lag_days`, que viene de `DataPolicy`.

Valores por defecto:

| Frecuencia | Lag |
|---|---:|
| diaria | 1 dia |
| semanal | 3 dias |
| mensual | 15 dias |
| trimestral | 45 dias |
| anual | 90 dias |

Formula:

```text
freshness = 1 - avg_publication_lag_days / 90
```

Ejemplos:

| Lag Promedio | Freshness |
|---:|---:|
| 15 dias | 0.83 |
| 45 dias | 0.50 |
| 90 dias | 0.00 |

Interpretacion:

```text
freshness alta -> los datos son recientes
freshness baja -> el diagnostico depende de informacion rezagada
```

### `narrative_coherence`

`narrative_coherence` mide si macro tradicional y liquidez cuentan una historia
compatible.

Hay contradiccion cuando:

```text
abs(macro_score - liquidity_score) >= contradiction_threshold
y
sign(macro_score) != sign(liquidity_score)
```

Con `contradiction_threshold = 0.50`:

```text
macro_score = +0.80
liquidity_score = -0.60
diferencia = 1.40
signos opuestos = si
contradiccion = si
```

Entonces:

```text
narrative_coherence = 0.55
```

Si no hay contradiccion:

```text
narrative_coherence = 1.00
```

Interpretacion:

| Macro | Liquidez | Coherencia |
|---|---|---|
| positivo | positivo | Alta. |
| negativo | negativo | Alta, aunque el entorno sea malo. |
| positivo | negativo | Baja: crecimiento con plomeria restrictiva. |
| negativo | positivo | Baja: economia debil con liquidez favorable. |

La contradiccion no cambia automaticamente el regimen. Penaliza la confianza con
la que se acepta ese diagnostico.

### Como Influye En Decisiones

La confianza entra en tres lugares:

1. `CurrentRegimeSnapshot.confidence`: calidad del diagnostico actual.
2. `RegimeClassifier`: reduce la confianza del regimen esperado cuando los
   errores del forecast son altos.
3. `ScenarioEngine`: sube la probabilidad del escenario base cuando la confianza
   es alta y la reduce cuando la confianza es baja.

Lectura operativa:

```text
confidence alta -> se puede usar el escenario base con mayor conviccion
confidence baja -> revisar upside/downside, contribuidores y overrides
```

## `ForecastPolicy`

### Que Mide

`ForecastPolicy` gobierna como se proyectan los scores hacia adelante.

El forecast actual no pronostica cada serie individual. Pronostica columnas de
score:

```text
growth_score
inflation_score
labor_score
macro_score
liquidity_score
...
```

Esta decision reduce ruido y mantiene el output conectado con el objetivo:
identificar regimenes macro/liquidez, no predecir cada dato macro con precision
puntual.

### Horizonte

La politica actual usa:

```text
horizons_months = (3, 12, 6)
base_horizon_months = 3
strategic_horizon_months = 12
fallback_horizon_months = 6
```

Lectura:

| Horizonte | Uso |
|---:|---|
| 3 meses | Horizonte base, alineado con rebalanceo trimestral. |
| 12 meses | Horizonte estrategico, alineado con revision anual del IPS. |
| 6 meses | Horizonte intermedio de contraste. |

### Modelo Implementado Hoy

El modelo implementado es AR(1):

```text
y_t = alpha + phi * y_(t-1) + error_t
```

Para cada columna de score:

1. ordena la serie por fecha;
2. toma `y_t` y `y_(t-1)`;
3. estima `alpha` y `phi` por minimos cuadrados;
4. proyecta recursivamente 3, 12 y 6 meses;
5. calcula intervalos con volatilidad EWMA de residuos.

Formula recursiva:

```text
forecast_1 = alpha + phi * last_value
forecast_2 = alpha + phi * forecast_1
...
forecast_h = alpha + phi * forecast_(h-1)
```

Intervalo:

```text
interval = z * ewma_residual_volatility * sqrt(horizon)
lower = forecast - interval
upper = forecast + interval
```

Con `confidence_interval_z = 1.64`, la banda es una aproximacion tipo 90%.

### Diagnosticos Guardados

El forecast produce:

| Campo | Significado |
|---|---|
| `observations` | numero de observaciones usadas. |
| `intercept` | alpha del AR(1). |
| `phi` | persistencia del score. |
| `rmse` | error cuadratico medio del ajuste. |
| `mae` | error absoluto medio. |
| `residual_std` | desviacion estandar de residuos. |
| `ewma_residual_volatility` | volatilidad reciente de residuos. |
| `aic`, `bic` | criterios de informacion aproximados. |
| `acf_lag_1`, `acf_lag_3`, `acf_lag_6`, `acf_lag_12` | autocorrelaciones. |

Estos diagnosticos se escriben en:

```text
data/research_runs/<run_id>/model_diagnostics/forecast_diagnostics.csv
```

cuando `write_audit=True`.

## Auto SARIMA Frente A AR(1)

Auto SARIMA no esta implementado hoy. La forma correcta de incorporarlo no es
cambiar un parametro y asumir que funciona. Debe entrar como candidato contra el
benchmark AR(1).

Diseno propuesto:

```text
score series
  -> fit AR(1)
  -> fit Auto SARIMA candidates
  -> rolling backtest
  -> comparar RMSE, MAE, direccion y BIC
  -> aceptar Auto SARIMA solo si vence AR(1)
  -> guardar decision y diagnosticos
```

La politica podria ampliarse asi:

```python
ForecastPolicy(
    benchmark_model="ar1",
    candidate_models=("ar1", "auto_sarima"),
    selection_metric="rolling_rmse",
    min_improvement=0.05,
    max_order=(3, 1, 3),
    seasonal_periods=(12,),
)
```

Regla de aceptacion:

```text
si auto_sarima_rmse <= ar1_rmse * (1 - min_improvement):
    aceptar Auto SARIMA
si no:
    usar AR(1)
```

Ejemplo de reporte:

```text
Target: macro_score
Benchmark: AR(1)
Candidate: Auto SARIMA
AR(1) rolling RMSE: 0.42
Auto SARIMA rolling RMSE: 0.37
Improvement: 11.9%
Decision: Auto SARIMA accepted
```

Si no mejora:

```text
Decision: Auto SARIMA rejected
Reason: did not outperform AR(1) by required threshold
```

Este diseno mantiene el principio metodologico: un modelo mas complejo solo se
acepta si vence un benchmark simple, auditable y dificil de justificar menos.

## `ScenarioPolicy`

### Que Hace

`ScenarioPolicy` convierte el regimen esperado en escenarios accionables.

Accionable no significa "comprar X". Significa que el escenario puede cambiar:

- tilts de asset class;
- preferencia por estilos;
- exposicion a credito;
- duracion;
- cash buffer;
- controles de liquidez;
- severidad del handoff a selection.

### Formula Implementada

Primero calcula probabilidades mecanicas:

```text
base = base_floor + base_confidence_slope * confidence

downside = downside_base - downside_confidence_penalty * confidence

si current_liquidity_regime in {restrictiva, estresada}:
    downside += liquidity_stress_penalty

upside = max(0.05, 1 - base - downside)
```

Despues normaliza:

```text
base + upside + downside = 1
```

Con defaults:

```text
base_floor = 0.40
base_confidence_slope = 0.30
downside_base = 0.35
downside_confidence_penalty = 0.15
liquidity_stress_penalty = 0.10
```

Ejemplo:

```text
confidence = 0.70
liquidity_regime actual = restrictiva

base = 0.40 + 0.30 * 0.70 = 0.61
downside = 0.35 - 0.15 * 0.70 = 0.245
downside = 0.245 + 0.10 = 0.345
upside = max(0.05, 1 - 0.61 - 0.345) = 0.05

total = 1.005

base_normalizado = 0.607
downside_normalizado = 0.343
upside_normalizado = 0.050
```

Interpretacion:

```text
Mayor confianza -> mas peso al escenario base.
Mayor confianza -> menos peso al downside.
Liquidez restrictiva o estresada -> penalizacion adicional al downside.
Upside nunca cae por debajo de 5% para evitar falsa certeza.
```

### De Donde Sale Cada Escenario

| Escenario | Regimen Usado | Fuente |
|---|---|---|
| `base` | regimen esperado central | forecast central del horizonte base. |
| `upside` | regimen del limite superior | banda superior del forecast. |
| `downside` | regimen del limite inferior | banda inferior del forecast. |

En otras palabras:

```text
base = trayectoria central
upside = trayectoria mejor que la central
downside = trayectoria peor que la central
```

### Donde Entra El Humano

El humano no debe editar silenciosamente probabilidades. Debe usar
`ScenarioOverride`:

```python
from src.research.macro_liquidity import ScenarioOverride

overrides = [
    ScenarioOverride(
        scenario_name="downside",
        probability=0.35,
        rationale="Credit spreads and funding stress deteriorated after the model cutoff.",
        approved_by="investment_committee",
    )
]
```

El override se guarda en:

```text
data/research_runs/<run_id>/scenario_overrides.csv
```

## Lectura Operativa En Notebook

Despues de correr:

```python
result = workflow.run(...)
```

usar:

```python
# Calidad del diagnostico actual.
result.nowcast.confidence_components.T
```

```python
# Senales que mas explican el diagnostico.
result.nowcast.contributors.head(20)
```

```python
# Forecast por target, horizonte e intervalo.
result.forecast.forecasts
```

```python
# Diagnosticos del AR(1).
result.forecast.diagnostics
```

```python
# Escenarios y probabilidades.
result.scenarios.scenario_table
```

```python
# Overrides humanos aplicados.
result.scenarios.override_log
```

```python
# Output que puede alimentar selection.
result.selection_context.to_frame()
```

## Riesgos Y Limites

| Riesgo | Explicacion | Mitigacion |
|---|---|---|
| Confianza mal interpretada | Puede confundirse con probabilidad de acierto. | Reportar como indice de calidad de evidencia. |
| `agreement` alto con senales negativas | Agreement mide alineacion, no direccion positiva. | Leer junto con contributors y scores. |
| AR(1) demasiado simple | Puede no capturar estacionalidad o cambios de regimen. | Agregar ModelSelector y Auto SARIMA contra benchmark. |
| ScenarioPolicy mecanico | Probabilidades iniciales son reglas, no posterior bayesiano. | Permitir overrides auditados y revisar en comite. |
| Interpolacion de trimestrales | Suaviza series como GDP y puede ocultar saltos. | Documentar `quarterly_interpolation=linear` en cada run. |
| Datos revisados | No replican informacion disponible en tiempo real. | Reportar `data_mode=revised`; usar vintages en fase robusta. |

## Checklist De Revision

Antes de aceptar la postura macro/liquidez:

- [ ] Revisar `result.nowcast.confidence_components.T`.
- [ ] Confirmar que `confidence` no se esta leyendo como probabilidad de acierto.
- [ ] Revisar `result.nowcast.contributors.head(20)`.
- [ ] Revisar si `macro_liquidity_contradiction` es `True`.
- [ ] Revisar `result.forecast.diagnostics`.
- [ ] Confirmar si AR(1) es suficiente para el uso actual.
- [ ] Revisar escenarios y probabilidades.
- [ ] Documentar cualquier `ScenarioOverride`.
- [ ] Revisar `asset_view` antes de pasar a selection.
- [ ] Guardar auditoria bajo `data/research_runs/<run_id>/`.

