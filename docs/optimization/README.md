# Optimization Module

El módulo `src.optimization` concentra la lógica de asignación óptima de pesos
para portafolios. Su papel dentro del proyecto es convertir un conjunto de
activos, precios históricos y supuestos de riesgo-retorno en una decisión de
capital: cuánto invertir en cada activo bajo una función objetivo explícita.

En términos conceptuales, el módulo responde preguntas como:

- ¿Cuál es la combinación de activos con menor varianza anualizada?
- ¿Qué pesos maximizan la relación retorno-riesgo medida por Sharpe?
- ¿Cómo cambia la asignación cuando el riesgo se mide solo por caídas debajo de
  un umbral?
- ¿Qué portafolio favorece activos con mejor comportamiento relativo entre
  potencial alcista y riesgo de pérdida?

La implementación cubre dos familias de optimización:

- **Media-varianza**, basada en la teoría moderna de portafolios de Markowitz.
- **Post-moderna o downside**, basada en semivarianza, downside risk y Omega.

## Estructura

```text
src/optimization/
├── __init__.py
├── configs.py
├── mean_variance.py
├── postmodern.py
└── results.py
```

## Documentación

- [architecture.md](architecture.md): arquitectura, capas, relaciones entre
  clases y mecanismos internos.
- [api_reference.md](api_reference.md): parámetros, atributos, métodos y
  retornos de la API pública.
- [workflows.md](workflows.md): ejemplos prácticos de implementación en flujos
  de investigación, construcción de portafolio y backtesting.
- [diagrams/README.md](diagrams/README.md): esquemas visuales del módulo con
  vistas de arquitectura, clases/componentes y flujo orgánico de uso.

## Diagramas

La carpeta [diagrams](diagrams/README.md) contiene el paquete visual del módulo
siguiendo la estructura de `diagram-architect`:

- [optimization_module_architecture.drawio](diagrams/optimization_module_architecture.drawio):
  vista universo de entradas, rutas de API, objetivos, solver, resultados y
  handoff downstream.
- [optimization_class_architecture.drawio](diagrams/optimization_class_architecture.drawio):
  vista de clases, funciones, estado, contratos de configuración y resultados.
- [optimization_workflow.drawio](diagrams/optimization_workflow.drawio): flujo
  operativo desde mandato y datos preparados hasta pesos optimizados y feedback.

También hay enlaces editables directos en
[open_in_diagrams_net.md](diagrams/open_in_diagrams_net.md).

## Flujo Orgánico

El módulo se ubica después de la selección y antes del backtesting o del análisis
de riesgo:

```text
selection -> portfolio -> optimization -> backtesting -> risk
```

Un uso típico parte de un objeto `Portfolio`, pasa por un optimizador y termina
con pesos actualizados:

```python
from src.portfolio import Portfolio
from src.optimization import MeanVarianceOptimizer, MinimumVarianceConfig

portfolio = Portfolio(
    prices=prices,
    weights=[1 / len(prices.columns)] * len(prices.columns),
)

optimizer = MeanVarianceOptimizer(portfolio=portfolio)
result = optimizer.optimize_minimum_variance(
    MinimumVarianceConfig(minimum_return=0.08),
)

result.weights_by_ticker
```

Si la optimización es exitosa, el optimizador actualiza el objeto `Portfolio` en
memoria. Esto permite que los módulos posteriores reutilicen el portafolio ya
optimizado sin reconstruir manualmente los pesos.

## Dos APIs Que Conviven

El módulo conserva una API histórica y ofrece una API nueva por composición.
Ambas conviven porque los notebooks del curso y la arquitectura más reciente no
parten del mismo punto.

La API histórica recibe tickers y fechas:

```python
from src.optimization import PortfolioOptimization

legacy_optimizer = PortfolioOptimization(
    tickers=["AAPL", "MSFT", "NVDA"],
    start="2020-01-01",
    end="2025-01-01",
    weight=[1 / 3, 1 / 3, 1 / 3],
)
```

La API nueva recibe un `Portfolio` ya construido:

```python
from src.optimization import MeanVarianceOptimizer

optimizer = MeanVarianceOptimizer(portfolio=portfolio)
```

La segunda forma es la recomendada para código modular, backtesting y pruebas,
porque separa claramente los datos del portafolio, las métricas y el proceso de
optimización.

## Principios De Diseño

El diseño actual se apoya en cuatro ideas:

1. Separar configuración, datos, solver y resultados.
2. Mantener compatibilidad con notebooks existentes.
3. Usar objetos `Portfolio` como unidad natural de composición.
4. Devolver resultados estructurados y serializables, no solo vectores de pesos.

El usuario no necesita interactuar directamente con `scipy.optimize.minimize`.
Las clases públicas construyen funciones objetivo, restricciones, bounds y
estructuras de salida. Esto permite concentrarse en la decisión financiera:
elegir el objetivo, los límites de inversión, la tasa libre de riesgo, el umbral
downside o el retorno mínimo requerido.

## Convenciones

- Los pesos deben ser vectores unidimensionales y sumar 1.
- Si no se permiten cortos, los límites default son `[0, 1]` por activo.
- Si se permiten cortos, los límites default son `[-1, 1]` por activo.
- El solver default es `SLSQP`, apropiado para restricciones lineales y bounds.
- Los resultados incluyen tanto `weights` como `weights_by_ticker`.
- Los optimizadores por composición actualizan el `Portfolio` solo cuando
  `result.success` es verdadero.
