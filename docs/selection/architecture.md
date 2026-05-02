# Arquitectura Del Módulo `selection`

`src.selection` está pensado como una capa previa a la asignación de capital. No
decide pesos, no estima fronteras eficientes y no ejecuta backtests. Su trabajo
es reducir un universo inicial a un conjunto de candidatos con lógica explícita.

## Capas Del Módulo

```text
Yahoo Finance / precios de mercado
        │
        ▼
fundamentals.py
        │
        ▼
fundamental_metrics.py
        │
        ▼
fundamental_scorers.py
        │
        ▼
fundamental_selector.py
        │
        ▼
notebooks / optimization / backtesting / risk
```

La selección por correlación sigue un flujo paralelo:

```text
Yahoo Finance / precios históricos
        │
        ▼
correlation_selector.py
        │
        ▼
universo diversificado de activos
```

## Responsabilidades Por Archivo

### `fundamentals.py`

Contiene la capa de acceso a Yahoo Finance.

Responsabilidades:

- Crear objetos `FundamentalData`.
- Descargar estados financieros, información general y precios.
- Descargar estados anuales y trimestrales cuando Yahoo Finance los expone.
- Normalizar tickers.
- Mantener a `yfinance` encapsulado en una sola capa.
- Cachear registros para evitar repetir consultas sobre el mismo universo.

Clases principales:

- `FundamentalData`
- `YahooFundamentalsProvider`

### `fundamental_metrics.py`

Contiene cálculos financieros puros.

Responsabilidades:

- Transformar `FundamentalData` en métricas comparables.
- Calcular ratios como PER, PBV, ROE, márgenes, deuda y crecimiento.
- Construir historiales de métricas por trimestre o año.
- Calcular crecimiento periodo contra periodo y crecimiento anual comparable.
- Evitar llamadas directas a Yahoo Finance.

Funciones principales:

- `build_fundamental_metrics`
- `build_metrics_frame`

### `fundamental_scorers.py`

Contiene las reglas de puntuación.

Responsabilidades:

- Definir configuraciones para estrategias.
- Convertir métricas en percentiles comparables.
- Crear un `fundamental_score` ponderado.
- Puntuar compañías por periodo para observar la evolución del score.
- Separar las reglas de inversión del proceso de descarga.

Clases principales:

- `FundamentalScoreConfig`
- `ValueScoreConfig`
- `GrowthScoreConfig`

Funciones principales:

- `config_for_strategy`
- `score_fundamentals`

### `fundamental_selector.py`

Contiene la interfaz de alto nivel para usuarios.

Responsabilidades:

- Orquestar descarga, cálculo de métricas y ranking.
- Orquestar métricas históricas y ranking temporal.
- Construir reportes de interpretabilidad para los top seleccionados.
- Exponer métodos simples para notebooks.
- Guardar resultados intermedios para inspección posterior.

Clase principal:

- `FundamentalSelector`

### `correlation_selector.py`

Contiene la selección basada en correlación.

Responsabilidades:

- Descargar precios.
- Calcular retornos.
- Rankear activos con menor correlación promedio.
- Construir candidatos diversificados dentro y entre grupos.

Clases principales:

- `CorrelationPortfolioSelector`
- `CorrelationSelector`

## Relación Entre Clases

```text
FundamentalSelector
    ├── usa YahooFundamentalsProvider
    │       └── produce FundamentalData
    ├── usa build_metrics_frame
    │       └── produce métricas por ticker
    ├── usa build_metric_history_frame
    │       └── produce métricas por ticker y periodo
    └── usa score_fundamentals
            └── produce ranking final
```

En términos prácticos:

1. `FundamentalSelector.rank(tickers)` pide datos al proveedor.
2. El proveedor devuelve un diccionario de `FundamentalData`.
3. Las métricas se calculan con `build_metrics_frame`.
4. Las métricas se puntúan con `score_fundamentals`.
5. El usuario recibe un `DataFrame` ordenado por `fundamental_score`.

Para análisis temporal:

1. `FundamentalSelector.rank_over_time(...)` pide datos al mismo proveedor.
2. El proveedor reutiliza registros cacheados si ya existen.
3. `build_metric_history_frame` construye métricas por trimestre o año.
4. `score_fundamentals_over_time` calcula percentiles dentro de cada periodo.
5. El usuario recibe un historial de score por ticker y fecha de reporte.

## Encaje Con El Proyecto

El módulo encaja en el flujo general del repositorio de esta manera:

```text
selection
    selecciona candidatos

research
    analiza comportamiento histórico o hipótesis

optimization
    calcula pesos óptimos

backtesting
    evalúa reglas de inversión

risk
    mide exposición, volatilidad, drawdowns y riesgo relativo
```

La salida natural de `selection` es una lista de tickers:

```python
selected_tickers = selected["ticker"].tolist()
```

Esa lista se puede pasar a los demás módulos sin adaptar estructuras complejas.

## Decisiones De Diseño

### Separación Entre Datos Y Scoring

Yahoo Finance es una fuente externa. Puede devolver campos incompletos, cambios
de nombre o datos no comparables entre compañías. Por eso la capa de datos no
decide qué compañía es mejor. Solo descarga y empaqueta.

El scoring se hace después, sobre `DataFrame`s. Esto permite:

- Probar reglas con datos simulados.
- Reutilizar métricas con fuentes distintas en el futuro.
- Cambiar ponderaciones sin tocar la descarga.

### Scores Cross-Sectionales

`score_fundamentals` usa rankings percentiles dentro del universo recibido. Esto
significa que el score no es absoluto: compara compañías entre sí.

Si el universo cambia, el score también puede cambiar. Esto es deseable para
selección de activos, porque normalmente se quiere saber cuáles son mejores
dentro de un conjunto de alternativas.

### Atributos Con Sufijo `_`

`FundamentalSelector` usa atributos como `metrics_` y `ranking_`. Esta convención
es común en librerías de análisis como scikit-learn: indica que el atributo se
crea después de ejecutar un método de cómputo.

```python
selector = FundamentalSelector(strategy="value")
ranking = selector.rank(tickers)

selector.metrics_
selector.ranking_
```

También existen:

```python
selector.metric_history_
selector.ranking_history_
```

Estos atributos se crean después de ejecutar `collect_metric_history` o
`rank_over_time`.

### Caché Del Proveedor

`YahooFundamentalsProvider` guarda en memoria los registros descargados. La clave
de caché considera:

- ticker
- fecha inicial
- fecha final
- campo de precio

Esto evita descargas repetidas cuando se ejecutan flujos como:

```python
selector.collect_metrics(tickers)
selector.rank_over_time(tickers)
selector.selection_report(top_k=5)
```

Si se desea forzar una nueva descarga:

```python
selector.provider.clear_cache()
```
