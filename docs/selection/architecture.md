# Arquitectura Del Módulo `selection`

## Índice Del Documento

1. Lectura Arquitectónica
2. Correspondencia Con Diagramas
3. Responsabilidades Por Archivo
4. Relaciones Principales
5. Contrato De Salida

## Mapa De La Serie Piloto

La ruta nueva empieza en [00_index.md](00_index.md). Este documento se conserva
como referencia arquitectónica heredada y se relaciona con
[02_architecture.md](02_architecture.md).

`src.selection` es la primera etapa del flujo de portfolio management. Su
responsabilidad es reducir un universo amplio de activos a una lista corta,
explicable y trazable de candidatos. No decide pesos, no construye órdenes, no
simula ejecución y no calcula reportes finales de riesgo.

```text
selection -> optimization -> portfolio construction -> execution/backtesting -> performance analysis -> risk/reporting
```

El contrato de salida natural es:

```text
tickers_finales + evidencia de selección
```

Esa evidencia puede venir de una tesis fundamental, de una lectura de
diversificación por correlación o de una combinación de ambas.

## Lectura Arquitectónica

La arquitectura del módulo se entiende mejor como un pipeline con bifurcación y
convergencia:

```text
mandato + universo + fechas
        │
        ▼
yfinance / Yahoo Finance
        │
        ▼
decisión metodológica
        ├── ruta fundamental: datos -> métricas -> scoring -> top-k -> reporte
        └── ruta correlación: precios -> retornos -> matriz -> ranking -> candidatos
        │
        ▼
lista corta auditable para optimization
```

La bifurcación responde a dos preguntas distintas:

- Ruta fundamental: qué compañías son más atractivas bajo una tesis `value` o
  `growth`.
- Ruta de correlación: qué activos reducen redundancia estadística dentro o
  entre grupos.

La convergencia existe porque ambas rutas terminan en el mismo tipo de decisión:
un conjunto de tickers candidatos para el siguiente módulo.

## Correspondencia Con Diagramas

La carpeta [diagrams](diagrams/README.md) contiene la vista visual canónica de
esta arquitectura:

1. [selection_module_architecture.drawio](diagrams/selection_module_architecture.drawio)
   Vista universo: archivos, responsabilidades, bifurcación fundamental vs
   correlación y contrato hacia `optimization`.

2. [selection_class_architecture.drawio](diagrams/selection_class_architecture.drawio)
   Vista de clases, funciones, atributos, métodos y conceptos de cálculo.

3. [selection_workflow.drawio](diagrams/selection_workflow.drawio)
   Vista orgánica de uso: mandato, universo, decisión metodológica, gates,
   consolidación, handoff y feedback.

La relación entre los tres diagramas es intencional:

```text
arquitectura del módulo -> arquitectura de clases -> flujo orgánico de uso
```

El primer diagrama es el universo; el segundo reduce el foco a clases y
funciones; el tercero reduce el foco al uso real dentro de un proceso de
portfolio management.

## Responsabilidades Por Archivo

### `__init__.py`

Define la API pública del paquete `src.selection`.

Expone:

- `FundamentalSelector`
- `CorrelationPortfolioSelector`
- `CorrelationSelector`
- `YahooFundamentalsProvider`
- `FundamentalData`
- `FundamentalScoreConfig`
- `MetricSignalSpec`
- `FUNDAMENTAL_METRIC_SIGNAL_SPECS`
- `fundamental_metric_signal_specs`
- `ValueScoreConfig`
- `GrowthScoreConfig`
- builders de métricas
- funciones de scoring

### `fundamentals.py`

Contiene la capa de acceso a datos fundamentales.

Responsabilidades:

- Normalizar tickers.
- Encapsular `yf.Ticker`.
- Descargar `info`, estados financieros, balances, cash flow y precios.
- Construir objetos `FundamentalData`.
- Cachear objetos de ticker y registros descargados por ticker, fechas y campo
  de precio.

Clases principales:

- `FundamentalData`
- `YahooFundamentalsProvider`

### `fundamental_metrics.py`

Contiene cálculos financieros puros sobre `FundamentalData`.

Responsabilidades:

- Transformar estados financieros en métricas comparables.
- Calcular valuación, rentabilidad, liquidez, leverage, cash flow, márgenes y
  crecimiento.
- Construir métricas de snapshot actual y métricas históricas por periodo.
- Mantener la lógica de cálculo separada de la descarga de datos.

Funciones principales:

- `build_fundamental_metrics`
- `build_metrics_frame`
- `build_fundamental_metric_history`
- `build_metric_history_frame`

### `fundamental_scorers.py`

Contiene la tesis de inversión como configuración y el cálculo de score.

Responsabilidades:

- Definir pesos y dirección de cada métrica.
- Definir señales granulares por métrica: nivel, cambio reciente, cambio
  interanual y cambio histórico promedio.
- Separar estrategias `value` y `growth`.
- Transformar métricas heterogéneas en percentiles cross-sectionales robustos.
- Calcular `fundamental_score` en escala 0-100.
- Calcular scores dentro de cada periodo para análisis temporal.

Clases principales:

- `FundamentalScoreConfig`
- `MetricSignalSpec`
- `ValueScoreConfig`
- `GrowthScoreConfig`

Funciones principales:

- `config_for_strategy`
- `fundamental_metric_signal_specs`
- `score_fundamentals`
- `score_fundamentals_over_time`

### `fundamental_selector.py`

Contiene la interfaz de alto nivel para workflows y notebooks.

Responsabilidades:

- Orquestar proveedor, métricas y scoring.
- Ejecutar ranking actual y ranking temporal.
- Seleccionar top-k.
- Construir reportes interpretativos.
- Guardar estados intermedios para auditoría.

Clase principal:

- `FundamentalSelector`

Estados auditables:

- `raw_data`
- `metrics_`
- `metric_history_`
- `ranking_`
- `ranking_history_`

### `correlation_selector.py`

Contiene la ruta de diversificación estadística.

Responsabilidades:

- Descargar precios con `yf.download`.
- Filtrar activos por cobertura mínima de datos.
- Calcular retornos.
- Construir matrices de correlación.
- Rankear activos por menor correlación media.
- Construir candidatos multi-grupo.
- Sugerir activos nuevos frente a un portafolio existente.

Clase principal:

- `CorrelationPortfolioSelector`

Alias público:

- `CorrelationSelector`

## Relaciones Principales

### Ruta Fundamental

```text
FundamentalSelector
    ├── usa YahooFundamentalsProvider
    │       └── produce FundamentalData
    ├── usa build_metrics_frame
    │       └── produce metrics_ por ticker
    ├── usa build_metric_history_frame
    │       └── produce metric_history_ por ticker y periodo
    ├── usa score_fundamentals
    │       └── produce ranking_
    └── usa score_fundamentals_over_time
            └── produce ranking_history_
```

Flujo operativo:

1. `rank(tickers)` normaliza el universo.
2. `provider.fetch_many(...)` devuelve `dict[str, FundamentalData]`.
3. `build_metrics_frame(...)` produce métricas comparables.
4. `score_fundamentals(...)` calcula percentiles y score compuesto.
5. `select_top(...)` devuelve el subconjunto superior.
6. `selection_report(...)` explica pesos, componentes y métricas usadas.

### Ruta Temporal Fundamental

```text
rank_over_time(...)
    ├── collect_metric_history(...)
    ├── build_metric_history_frame(...)
    ├── score_fundamentals_over_time(...)
    └── metric_evolution(...)
```

Esta ruta permite revisar si el score y las métricas mejoran, se deterioran o
son inestables a través de trimestres o años.

### Ruta De Correlación

```text
CorrelationPortfolioSelector
    ├── _download_prices(...)
    ├── _apply_coverage_filter(...)
    ├── returns = prices.pct_change(...)
    ├── corr = returns.corr()
    ├── _compute_corr_scores(...)
    ├── rank_within_groups(...)
    ├── build_multigroup_portfolio(...)
    └── update_portfolio(...)
```

Esta ruta no usa `YahooFundamentalsProvider`. Ambos caminos comparten la fuente
externa `yfinance`, pero mantienen responsabilidades separadas:

- La ruta fundamental usa `yf.Ticker` a través de `YahooFundamentalsProvider`.
- La ruta de correlación usa `yf.download` directamente para precios.

## Contratos De Salida

La salida fundamental típica es:

```python
result = selector.run_pipeline(tickers, top_k=5)
selected_tickers = result["selected_tickers"]
report = result["report"]
```

La salida por correlación típica es:

```python
result = correlation_selector.run_pipeline(grouped_tickers)
final_tickers = result["final_tickers"]
candidate_corr_matrix = result["candidate_corr_matrix"]
```

En ambos casos, `selection` entrega candidatos explicables. La decisión de
pesos, objetivo, constraints y solver corresponde a `optimization`.

## Decisiones De Diseño

### Separación Entre Datos, Cálculo Y Tesis

Yahoo Finance puede devolver campos incompletos, nombres variables o series con
cobertura desigual. Por eso la arquitectura separa:

1. Adquisición de datos.
2. Transformación a métricas.
3. Scoring bajo una tesis.
4. Selección y explicación.

Esta separación permite probar métricas con datos simulados, cambiar pesos sin
tocar descargas y reemplazar la fuente de datos en el futuro.

### Scores Cross-Sectionales Y Señales

`score_fundamentals` calcula componentes normalizados dentro del universo
recibido. El score no es absoluto; depende del grupo comparado. Si cambia el
universo, puede cambiar el ranking.

La configuración nueva usa `MetricSignalSpec` para separar tres preguntas:

- cuál es el nivel más reciente de la métrica;
- cuánto cambió contra el periodo anterior o contra el mismo periodo del año
  anterior;
- cuál ha sido el cambio promedio en toda la ventana histórica disponible.

Cada componente se winsoriza y se transforma a un score comparable antes de
ponderarse. La salida incluye `score_coverage` para auditar qué proporción del
peso configurado tuvo datos disponibles por compañía.

Esto es coherente con selección de activos: normalmente interesa comparar
alternativas dentro de un conjunto candidato.

### Estados Con Sufijo `_`

`FundamentalSelector` usa atributos como `metrics_`, `ranking_`,
`metric_history_` y `ranking_history_`. El sufijo indica que el atributo se
crea después de ejecutar un método de cómputo.

```python
selector = FundamentalSelector(strategy="value")
ranking = selector.rank(tickers)

selector.metrics_
selector.ranking_
```

### Caché Del Proveedor

`YahooFundamentalsProvider` guarda en memoria objetos de ticker y registros
descargados. La clave de caché considera:

- ticker
- fecha inicial
- fecha final
- campo de precio

Si se desea forzar una nueva descarga:

```python
selector.provider.clear_cache()
```
