# Diagramas Del Modulo `selection`

## Índice Del Documento

1. Lectura De Los Diagramas
2. Archivos Drawio
3. Enlaces Directos A Diagrams.net
4. Coherencia Con La Documentacion
5. Exportaciones A Imagen

## Mapa De La Serie Piloto

La ruta documental nueva empieza en [../00_index.md](../00_index.md). La política
general de diagramas vive en [../../_meta/04_diagram_policy.md](../../_meta/04_diagram_policy.md).

Estos entregables modelan `src/selection` como la primera etapa del flujo de
portfolio management:

```text
selection -> optimization -> portfolio construction -> execution/backtesting -> performance analysis -> risk/reporting
```

La lectura sugerida va de mayor a menor escala y cada diagrama esta
estructurado como pipeline lineal. La idea es que la vista se lea de izquierda a
derecha o de arriba hacia abajo, con bifurcaciones explicitas cuando el flujo
separa la ruta fundamental de la ruta de correlacion:

1. [selection_module_architecture.drawio](./selection_module_architecture.drawio)  
   Vista universo del modulo: archivos, responsabilidades, bifurcaciones,
   convergencias y lugar dentro del flujo general.

2. [selection_class_architecture.drawio](./selection_class_architecture.drawio)  
   Vista de clases, funciones, atributos, metodos, relaciones y conceptos de
   calculo que sostiene cada objeto, ordenada como una cadena operativa.

3. [selection_workflow.drawio](./selection_workflow.drawio)  
   Vista organica de uso en un proceso real: desde universo inicial hasta
   lista de candidatos para optimizacion, incluyendo compuertas de revision y
   feedback posterior.

Los archivos `.drawio` se pueden abrir directamente en diagrams.net con
`File -> Open From -> Device`.

Tambien puedes abrir copias editables directamente en diagrams.net desde la
pagina local de apertura:

- [open_in_diagrams_net.md](./open_in_diagrams_net.md)

## Coherencia Con La Documentacion

Estos diagramas son la vista visual canónica de
[architecture.md](../architecture.md), [api_reference.md](../api_reference.md)
y [workflows.md](../workflows.md). La consistencia esperada es:

- La arquitectura del módulo explica la bifurcación fundamental/correlación y
  su convergencia hacia `optimization`.
- La arquitectura de clases muestra que `YahooFundamentalsProvider` pertenece a
  la ruta fundamental; `CorrelationPortfolioSelector` usa `yfinance` de forma
  directa para precios, sin depender del proveedor fundamental.
- El flujo orgánico muestra los gates de revisión y el handoff hacia el
  siguiente módulo, no una asignación final de pesos.

## Exportaciones A Imagen

La carpeta [exports](exports/README.md) reserva el lugar para imágenes `.png` o
`.svg`. En este piloto las imágenes están pendientes porque el entorno actual no
tiene una herramienta CLI de diagrams.net disponible.
