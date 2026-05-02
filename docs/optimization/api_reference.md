# Referencia API

Esta referencia describe la API pública de `src.optimization`.

## Configuraciones

### `OptimizationConfig`

```python
class OptimizationConfig(
    risk_free_rate=0.0,
    allow_short=False,
    bounds=None,
    initial_weights=None,
    solver_method="SLSQP",
    solver_options={"maxiter": 500, "ftol": 1e-9, "disp": False},
)
```

Configuración compartida para objetivos de media-varianza.

#### Parámetros

`risk_free_rate` : `float`, default `0.0`  
Tasa libre de riesgo anual usada para calcular Sharpe.

`allow_short` : `bool`, default `False`  
Si es `True`, permite pesos negativos cuando no se entregan `bounds`
explícitos.

`bounds` : sequence of `(float, float)`, optional  
Límites por activo. Debe tener la misma longitud que el número de activos.

`initial_weights` : iterable of `float`, optional  
Punto inicial para el solver. Debe ser un vector unidimensional, sumar 1 y
respetar los bounds.

`solver_method` : `str`, default `"SLSQP"`  
Método enviado a `scipy.optimize.minimize`.

`solver_options` : `dict`, optional  
Opciones del solver. Por default se usan 500 iteraciones máximas, tolerancia
`1e-9` y salida silenciosa.

#### Métodos

`to_legacy()`  
Devuelve la misma configuración. Existe para mantener una interfaz compatible
con la migración desde la capa histórica.

### `MinimumVarianceConfig`

```python
class MinimumVarianceConfig(OptimizationConfig, minimum_return=None)
```

Configuración para mínima varianza.

#### Parámetros Adicionales

`minimum_return` : `float`, optional  
Retorno anual mínimo requerido. Si se define, el problema agrega la restricción
`w @ mu >= minimum_return`.

### `PostModernOptimizationConfig`

```python
class PostModernOptimizationConfig(
    threshold=0.0,
    allow_short=False,
    bounds=None,
    initial_weights=None,
    solver_method="SLSQP",
    solver_options={"maxiter": 500, "ftol": 1e-9, "disp": False},
)
```

Configuración compartida para objetivos downside.

#### Parámetros

`threshold` : `float`, default `0.0`  
Referencia contra la que se separan retornos downside y upside.

`allow_short`, `bounds`, `initial_weights`, `solver_method`,
`solver_options`  
Tienen el mismo significado que en `OptimizationConfig`.

### `MinimumSemivarianceConfig`

```python
class MinimumSemivarianceConfig(
    PostModernOptimizationConfig,
    minimum_return=None,
)
```

Configuración para mínima semivarianza.

#### Parámetros Adicionales

`minimum_return` : `float`, optional  
Retorno anual mínimo requerido.

### `MaximumOmegaConfig`

```python
class MaximumOmegaConfig(PostModernOptimizationConfig)
```

Configuración para maximizar Omega. No agrega parámetros sobre
`PostModernOptimizationConfig`.

## Resultados

### `OptimizationResult`

Resultado de una optimización de media-varianza.

#### Atributos

`objective` : `str`  
Objetivo resuelto: `"minimum_variance"` o `"maximum_sharpe"`.

`success` : `bool`  
Indica si el solver reportó convergencia exitosa.

`status` : `int`  
Código de estado del solver.

`message` : `str`  
Mensaje de diagnóstico del solver.

`weights` : `numpy.ndarray`  
Vector de pesos óptimos.

`weights_by_ticker` : `pandas.Series`  
Pesos indexados por ticker.

`expected_return` : `float`  
Retorno esperado anual del portafolio óptimo.

`variance` : `float`  
Varianza anual del portafolio óptimo.

`volatility` : `float`  
Volatilidad anual del portafolio óptimo.

`sharpe` : `float`  
Sharpe anualizado usando `risk_free_rate`.

`objective_value` : `float`  
Valor reportado para el objetivo. En mínima varianza es la varianza; en máximo
Sharpe es el Sharpe positivo del resultado.

`iterations` : `int`  
Número de iteraciones reportadas por el solver.

#### Métodos

`from_legacy(result)`  
Construye un `OptimizationResult` desde cualquier objeto con los mismos
atributos.

### `PostModernOptimizationResult`

Resultado de una optimización downside.

#### Atributos

`objective` : `str`  
Objetivo resuelto: `"minimum_semivariance"` o `"maximum_omega"`.

`success`, `status`, `message`, `weights`, `weights_by_ticker`,
`expected_return`, `objective_value`, `iterations`  
Tienen la misma función que en `OptimizationResult`.

`semivariance` : `float`  
Semivarianza del portafolio óptimo.

`downside_risk` : `float`  
Raíz de la semivarianza del portafolio.

`omega` : `float`  
Omega del portafolio óptimo.

#### Métodos

`from_legacy(result)`  
Construye un `PostModernOptimizationResult` desde cualquier objeto compatible.

## Optimización Media-Varianza

### `PortfolioOptimization`

```python
class PortfolioOptimization(
    tickers,
    start,
    end=None,
    price_field="Close",
    weight=None,
)
```

Optimizador histórico basado en `AssetsResearch`. Es útil cuando se quiere
trabajar directamente con tickers y fechas.

#### Parámetros

`tickers` : iterable of `str`  
Activos del universo.

`start` : `str`  
Fecha inicial para la investigación de precios.

`end` : `str`, optional  
Fecha final exclusiva.

`price_field` : `str`, default `"Close"`  
Campo de precio usado por la capa de investigación.

`weight` : iterable of `float`  
Pesos iniciales. Es obligatorio y debe sumar 1.

#### Atributos

`weight` : `numpy.ndarray`  
Vector de pesos actual. Se actualiza después de una optimización exitosa.

`tickers`, `start`, `end`, `price_field`  
Atributos heredados de `AssetsResearch`.

#### Métodos De Métricas

`portfolio_path(weight=None)`  
Devuelve la serie de valor ponderado del portafolio.

`portfolio_annual_return(weight=None)`  
Devuelve el retorno anual esperado del portafolio.

`portfolio_variance(weight=None)`  
Devuelve la varianza anual del portafolio.

`portfolio_annual_volatility(weight=None)`  
Devuelve la volatilidad anual.

`portfolio_variance_coeficience(weight=None)`  
Devuelve volatilidad entre retorno esperado. El nombre conserva compatibilidad
con notebooks anteriores.

`portfolio_sharpe_ratio(free_rate, weight=None)`  
Devuelve Sharpe usando la tasa libre de riesgo recibida.

#### Métodos De Optimización

`optimize_minimum_variance(config=None)`  
Minimiza `w.T @ Sigma @ w`.

`optimize_maximum_sharpe(config=None)`  
Maximiza Sharpe minimizando su negativo.

### `MeanVarianceOptimizer`

```python
class MeanVarianceOptimizer(portfolio)
```

Adaptador recomendado para código modular. Recibe un `Portfolio` ya construido.

#### Parámetros

`portfolio` : `src.portfolio.Portfolio`  
Portafolio con precios, retornos, tickers y pesos iniciales.

#### Atributos

`portfolio` : `Portfolio`  
Objeto que será usado como fuente de datos y que será actualizado si la
optimización es exitosa.

#### Métodos

`optimize_minimum_variance(config=None)`  
Ejecuta mínima varianza y devuelve `OptimizationResult`.

`optimize_maximum_sharpe(config=None)`  
Ejecuta máximo Sharpe y devuelve `OptimizationResult`.

## Optimización Post-Moderna

### `PortfolioOptimizationPostModern`

```python
class PortfolioOptimizationPostModern(
    tickers,
    start,
    end=None,
    price_field="Close",
    weight=None,
)
```

Optimizador histórico para semivarianza y Omega.

#### Atributos Públicos De Estado

`returns_down` : `pandas.DataFrame` or `None`  
Última matriz de retornos debajo de la referencia.

`returns_up` : `pandas.DataFrame` or `None`  
Última matriz de retornos encima de la referencia.

`shortfall_risk` : `numpy.ndarray` or `None`  
Último vector de riesgo downside anualizado.

`upside_potential` : `numpy.ndarray` or `None`  
Último vector de riesgo upside anualizado.

#### Métodos De Separación Downside/Upside

`returns_below(threshold=0.0, benchmark_returns=None)`  
Devuelve retornos por debajo de la referencia.

`returns_above(threshold=0.0, benchmark_returns=None)`  
Devuelve retornos por encima de la referencia.

`downside_risk(threshold=None, benchmark_returns=None)`  
Devuelve riesgo downside anualizado por activo.

`upside_risk(threshold=None, benchmark_returns=None)`  
Devuelve riesgo upside anualizado por activo.

#### Métodos De Riesgo De Portafolio

`semivariance_matrix(threshold=None, benchmark_returns=None)`  
Construye la matriz de semivarianza.

`portfolio_semivariance(weight=None, threshold=None, benchmark_returns=None)`  
Calcula la semivarianza del portafolio.

`portfolio_downside_risk(weight=None, threshold=None, benchmark_returns=None)`  
Calcula la raíz de la semivarianza del portafolio.

`asset_omega_ratio(threshold=None, benchmark_returns=None)`  
Calcula Omega por activo.

`portfolio_omega_ratio(weight=None, threshold=None, benchmark_returns=None)`  
Calcula Omega del portafolio.

`semivarianza_down(threshold=None, benchmark_returns=None)`  
Alias compatible de `semivariance_matrix(...)`.

#### Métodos De Optimización

`optimize_minimum_semivariance(config=None, benchmark_returns=None)`  
Minimiza la semivarianza del portafolio.

`optimize_maximum_omega(config=None)`  
Maximiza Omega minimizando su negativo.

### `PostModernOptimizer`

```python
class PostModernOptimizer(portfolio)
```

Adaptador recomendado para objetivos downside cuando ya existe un `Portfolio`.

#### Parámetros

`portfolio` : `src.portfolio.Portfolio`

#### Métodos

`optimize_minimum_semivariance(config=None, benchmark_returns=None)`  
Ejecuta mínima semivarianza y devuelve `PostModernOptimizationResult`.

`optimize_maximum_omega(config=None)`  
Ejecuta máximo Omega y devuelve `PostModernOptimizationResult`.

## Alias Públicos

`PortfolioOptimizationPostMordern` es un alias con typo conservado para
compatibilidad con código antiguo. El nombre correcto es
`PortfolioOptimizationPostModern`.
