# Diagramas Del Modulo `optimization`

Estos entregables modelan `src/optimization` como la etapa que convierte un
universo de activos, precios historicos y supuestos de riesgo-retorno en una
asignacion auditable de pesos:

```text
selection -> portfolio / research -> optimization -> backtesting -> risk / reporting
```

La lectura sugerida va de mayor a menor escala. Cada diagrama esta construido
como una historia operativa: primero el universo del modulo, despues las clases
y contratos internos, y finalmente el flujo real de uso con decisiones,
validaciones, salidas y feedback.

1. [optimization_module_architecture.drawio](./optimization_module_architecture.drawio)  
   Vista universo del modulo: entradas, APIs de uso, archivos principales,
   objetivos, solver, resultados y handoff hacia backtesting o risk.

2. [optimization_class_architecture.drawio](./optimization_class_architecture.drawio)  
   Vista de clases, funciones, atributos, estado, caches, formulas y
   relaciones entre configuraciones, optimizadores y resultados.

3. [optimization_workflow.drawio](./optimization_workflow.drawio)  
   Vista organica de uso: desde mandato y datos preparados hasta seleccion de
   objetivo, validacion de restricciones, convergencia, actualizacion de pesos
   y aprendizaje posterior.

Los archivos `.drawio` se pueden abrir directamente en diagrams.net con
`File -> Open From -> Device`.

Tambien puedes abrir copias editables directamente en diagrams.net desde la
pagina local de apertura:

- [open_in_diagrams_net.md](./open_in_diagrams_net.md)

## Coherencia Con La Documentacion

Estos diagramas son la vista visual canonica de
[architecture.md](../architecture.md), [api_reference.md](../api_reference.md)
y [workflows.md](../workflows.md). La consistencia esperada es:

- La arquitectura del modulo explica la convivencia de la API historica basada
  en `AssetsResearch` y la API moderna por composicion sobre `Portfolio`.
- La arquitectura de clases muestra que el calculo principal vive en
  `PortfolioOptimization` y `PortfolioOptimizationPostModern`, mientras los
  adaptadores modernos construyen esos optimizadores e inyectan los datos del
  `Portfolio`.
- El flujo organico muestra que `optimization` decide pesos y devuelve
  resultados auditables; no selecciona tickers, no simula capital y no produce
  reportes finales.
