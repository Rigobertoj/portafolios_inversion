# Arquitectura Del Módulo `optimization`

`src.optimization` es la capa que transforma investigación histórica en una
decisión de asignación. Recibe precios, retornos y pesos iniciales; calcula
matrices o vectores de riesgo-retorno; arma un problema de optimización; llama a
un solver numérico; y devuelve un resultado interpretable por ticker.

El módulo tiene una arquitectura híbrida. Por un lado conserva clases históricas
basadas en `AssetsResearch`, útiles para notebooks antiguos. Por otro lado expone
adaptadores nuevos basados en composición con `Portfolio`, que son la pieza más
limpia para el resto del paquete.

## Capas Del Módulo

```text
precios / retornos históricos
        │
        ▼
Portfolio o AssetsResearch
        │
        ▼
PortfolioOptimization / PortfolioOptimizationPostModern
        │
        ▼
MeanVarianceOptimizer / PostModernOptimizer
        │
        ▼
OptimizationResult / PostModernOptimizationResult
        │
        ▼
backtesting / risk / notebooks / reportes
```

La ruta conceptual recomendada es:

```text
Portfolio
    contiene precios, retornos, tickers y pesos actuales

Optimizer
    construye un solver compatible con esos datos

Config
    define supuestos, restricciones y opciones numéricas

Result
    documenta el óptimo, métricas y estado de convergencia
```

## Responsabilidades Por Archivo

### `configs.py`

Contiene los objetos de configuración. Estas clases no optimizan ni calculan
métricas; solo describen las reglas del problema.

Clases principales:

- `OptimizationConfig`
- `MinimumVarianceConfig`
- `PostModernOptimizationConfig`
- `MinimumSemivarianceConfig`
- `MaximumOmegaConfig`

Responsabilidades:

- Definir `risk_free_rate` para Sharpe.
- Definir `threshold` para métricas downside.
- Controlar si se permiten ventas en corto con `allow_short`.
- Recibir `bounds` explícitos por activo.
- Recibir `initial_weights` cuando el usuario quiere un punto inicial distinto.
- Configurar `solver_method` y `solver_options`.
- Añadir restricciones específicas, como `minimum_return`.

### `results.py`

Contiene los objetos de salida. Estos objetos son importantes porque una
optimización no solo produce pesos: también produce un diagnóstico.

Clases principales:

- `OptimizationResult`
- `PostModernOptimizationResult`

Responsabilidades:

- Guardar `success`, `status`, `message` e `iterations`.
- Guardar pesos como `numpy.ndarray` y como `pandas.Series` indexado por ticker.
- Guardar métricas del portafolio óptimo.
- Proveer `from_legacy(...)` para traducir objetos con la misma forma a la capa
  nueva.

### `mean_variance.py`

Implementa la optimización clásica de media-varianza.

Clases principales:

- `PortfolioOptimization`
- `MeanVarianceOptimizer`

`PortfolioOptimization` es la clase compatible con la API histórica. Hereda de
`AssetsResearch`, por lo que puede descargar o reutilizar precios a partir de
tickers, fechas y un campo de precio. Además mantiene un vector interno
`weight`, validado para que sea unidimensional, tenga la misma longitud que los
tickers y sume 1.

`MeanVarianceOptimizer` es el adaptador moderno. Recibe un objeto `Portfolio`,
construye internamente un `PortfolioOptimization`, inyecta en caché los precios y
retornos ya existentes y ejecuta el objetivo solicitado. Si el solver converge,
actualiza los pesos del `Portfolio`.

Objetivos implementados:

- `optimize_minimum_variance(...)`
- `optimize_maximum_sharpe(...)`

### `postmodern.py`

Implementa objetivos downside o post-modernos.

Clases principales:

- `PortfolioOptimizationPostModern`
- `PostModernOptimizer`

`PortfolioOptimizationPostModern` extiende la lógica de media-varianza con
métricas que separan retornos por encima y por debajo de una referencia. Esa
referencia puede ser un umbral fijo (`threshold`) o un benchmark alineado en
fechas más el umbral.

`PostModernOptimizer` cumple el mismo papel que `MeanVarianceOptimizer`, pero
para objetivos downside.

Objetivos implementados:

- `optimize_minimum_semivariance(...)`
- `optimize_maximum_omega(...)`

## Relación Entre Clases

```text
Portfolio
    ├── prices
    ├── returns
    ├── tickers
    └── weights
          │
          ▼
MeanVarianceOptimizer
    └── construye PortfolioOptimization
            ├── calcula retornos esperados
            ├── calcula matriz de covarianza
            ├── resuelve con scipy.optimize.minimize
            └── produce OptimizationResult
```

Para objetivos post-modernos:

```text
Portfolio
    │
    ▼
PostModernOptimizer
    └── construye PortfolioOptimizationPostModern
            ├── calcula retornos ajustados por threshold o benchmark
            ├── separa retornos downside y upside
            ├── calcula semivarianza, downside risk u Omega
            ├── resuelve con scipy.optimize.minimize
            └── produce PostModernOptimizationResult
```

## Mecánica De Media-Varianza

La optimización de media-varianza usa dos insumos principales:

- Vector de retornos esperados anualizados, `mu`.
- Matriz de covarianza anualizada, `Sigma`.

El retorno esperado del portafolio se calcula como:

```text
E[R_p] = w @ mu
```

La varianza se calcula como:

```text
Var(R_p) = w.T @ Sigma @ w
```

La volatilidad anualizada es:

```text
Vol(R_p) = sqrt(Var(R_p))
```

El Sharpe se calcula como:

```text
Sharpe = (E[R_p] - risk_free_rate) / Vol(R_p)
```

Como `scipy.optimize.minimize` minimiza funciones, el máximo Sharpe se expresa
como la minimización del Sharpe negativo:

```text
objective(w) = -Sharpe(w)
```

## Mecánica Post-Moderna

La capa post-moderna cambia la forma de mirar el riesgo. En lugar de tratar
toda desviación como indeseable, separa retornos negativos y positivos respecto
a una referencia.

La referencia puede ser:

- `threshold`, por ejemplo 0.0 para medir retornos debajo de cero.
- `benchmark_returns + threshold`, cuando el riesgo se define contra un índice.

Primero se calculan desviaciones:

```text
deviation = asset_return - reference
```

Después se separan retornos:

```text
returns_below = deviation donde deviation < 0, si no 0
returns_above = deviation donde deviation > 0, si no 0
```

La semivarianza usa el riesgo downside anualizado de cada activo y la matriz de
correlación histórica para construir una matriz comparable a la covarianza:

```text
semivariance_matrix = outer(downside_risk, downside_risk) * correlation
```

El Omega por activo se aproxima como:

```text
omega = upside_risk / downside_risk
```

El Omega de portafolio se calcula como combinación ponderada:

```text
portfolio_omega = w @ omega_vector
```

## Restricciones Y Bounds

Todos los objetivos usan una restricción base:

```text
sum(w) = 1
```

Esto obliga a invertir el 100% del capital disponible.

Si el usuario agrega un retorno mínimo en `MinimumVarianceConfig` o
`MinimumSemivarianceConfig`, se añade:

```text
w @ mu >= minimum_return
```

Los bounds se resuelven así:

- Si `bounds` está definido, se usa exactamente ese arreglo y se valida su
  longitud.
- Si `bounds` no está definido y `allow_short=False`, cada peso queda entre 0 y
  1.
- Si `bounds` no está definido y `allow_short=True`, cada peso queda entre -1 y
  1.

## Estado Interno Y Caché

La ruta histórica usa cachés heredadas de `AssetsResearch` para precios y
retornos. La ruta por composición aprovecha eso de forma controlada: construye
un optimizador histórico y le inyecta los precios y retornos del `Portfolio`.
Así evita descargar datos dos veces y mantiene compatibilidad con métodos
preexistentes.

La clase post-moderna agrega cachés específicas por `threshold`:

- retornos debajo del umbral
- retornos encima del umbral
- downside risk
- upside risk
- matriz de semivarianza
- Omega por activo

Cuando se usa un benchmark externo, esos cálculos no se cachean de la misma
forma porque dependen de una serie adicional que puede cambiar en cada llamada.

## Actualización De Pesos

Hay una diferencia importante entre calcular un resultado y mutar el portafolio.

En la clase histórica:

```text
PortfolioOptimization.optimize_* -> actualiza self.weight si success=True
```

En la clase por composición:

```text
MeanVarianceOptimizer.optimize_* -> actualiza portfolio.weights si success=True
PostModernOptimizer.optimize_* -> actualiza portfolio.weights si success=True
```

Si el solver falla, los pesos originales se conservan. Esto evita que un
portafolio quede contaminado por una solución no válida.

## Encaje Con Backtesting

El módulo de backtesting no reimplementa optimización. Usa estrategias que
delegan aquí:

```text
MeanVarianceStrategy
    └── MeanVarianceOptimizer

PostModernStrategy
    └── PostModernOptimizer
```

En un backtest dinámico, este flujo se repite en cada fecha de rebalanceo:

1. El motor crea una ventana de precios in-sample.
2. La estrategia construye un `Portfolio` con pesos iniciales.
3. El optimizador calcula pesos óptimos.
4. El motor aplica esos pesos al siguiente tramo out-of-sample.
5. Se registra evolución, turnover, costos y métricas.

## Decisiones De Diseño

### Compatibilidad Sin Congelar La Arquitectura

El proyecto conserva `PortfolioOptimization` y
`PortfolioOptimizationPostModern` porque muchos notebooks trabajan directamente
con tickers y fechas. Sin embargo, la arquitectura más nueva prefiere `Portfolio`
como unidad explícita de datos. Por eso existen `MeanVarianceOptimizer` y
`PostModernOptimizer`: funcionan como puente entre una API cómoda para análisis
modular y una implementación que todavía conserva comportamiento histórico.

### Configuración Como Contrato

Los objetos `Config` hacen explícitas las decisiones del usuario. En vez de
pasar muchos argumentos sueltos a cada método, se agrupan supuestos de inversión
y del solver en una estructura legible. Esto facilita reutilizar la misma
configuración en investigación, backtesting estático y backtesting dinámico.

### Resultados Auditables

Los objetos `Result` evitan que la salida sea solo un vector anónimo. Cada
resultado dice qué objetivo se resolvió, si el solver tuvo éxito, cuál fue el
mensaje de convergencia, cuántas iteraciones tomó y qué métricas produjo el
portafolio óptimo. Eso permite explicar el proceso, no solo reportar pesos.

### Riesgo Como Elección De Modelo

Media-varianza y semivarianza no son versiones intercambiables del mismo
problema. Representan dos formas de definir riesgo. El módulo separa ambas
familias para que el usuario pueda elegir conscientemente si quiere castigar toda
volatilidad o solo desviaciones desfavorables.
