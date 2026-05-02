# Arquitectura De Optimización De Portafolios

La documentación vigente del módulo de optimización vive ahora en
[`docs/optimization/`](optimization/README.md).

Este archivo se conserva como punto de entrada para referencias anteriores, pero
la arquitectura actual ya no se describe correctamente como una cadena única de
herencia. El diseño vigente combina dos capas:

- Una capa histórica basada en `PortfolioOptimization` y
  `PortfolioOptimizationPostModern`, útil para notebooks que parten de tickers y
  fechas.
- Una capa por composición basada en `Portfolio`, `MeanVarianceOptimizer` y
  `PostModernOptimizer`, recomendada para código modular, backtesting y pruebas.

Para una explicación amplia y actualizada, consulta:

- [README del módulo](optimization/README.md)
- [Arquitectura](optimization/architecture.md)
- [Referencia API](optimization/api_reference.md)
- [Flujos de trabajo](optimization/workflows.md)
