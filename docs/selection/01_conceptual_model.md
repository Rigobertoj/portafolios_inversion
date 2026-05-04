# Selection: Modelo Conceptual

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Problema Que Resuelve
4. Qué No Resuelve
5. Dos Rutas De Selección
6. Contrato Conceptual De Salida
7. Relación Con Portfolio Management
8. Fuentes Conceptuales

## Mapa De La Serie

| Orden | Documento | Rol | Cuándo Leerlo |
|---:|---|---|---|
| 00 | `00_index.md` | Entrada secuencial | Siempre primero. |
| 01 | `01_conceptual_model.md` | Modelo conceptual | Antes de usar o modificar `selection`. |
| 02 | `02_architecture.md` | Arquitectura del módulo | Antes de tocar código. |
| 03 | `03_data_contracts.md` | Contratos de datos | Cuando se integran entradas o salidas. |
| 04 | `04_fundamental_selection_article.md` | Artículo fundamental | Para entender value, growth y scoring. |
| 05 | `05_correlation_selection_article.md` | Artículo correlación | Para entender diversificación estadística. |
| 06 | `06_scoring_model.md` | Modelo de score | Para ajustar métricas, señales o pesos. |
| 07 | `07_workflows.md` | Flujos prácticos | Para ejecutar casos de uso. |
| 08 | `08_api_reference.md` | API narrativa | Para consultar clases, métodos y atributos. |
| 09 | `09_validation_and_edge_cases.md` | Validación | Para pruebas, errores y límites. |
| 10 | `10_glossary.md` | Glosario | Para unificar vocabulario. |

## Propósito

`selection` es la etapa que reduce un universo amplio de activos a un conjunto
de candidatos razonado, trazable y apto para alimentar módulos posteriores como
`optimization`, `backtesting` o `risk`.

La selección no asigna pesos. Su salida principal no es una cartera final, sino
un universo candidato con evidencia.

## Problema Que Resuelve

En portfolio management rara vez se optimiza sobre todos los activos posibles.
Antes de construir pesos, el usuario necesita decidir qué activos merecen estar
en el universo de análisis.

El módulo responde:

```text
De un universo amplio, ¿qué activos pasan a la siguiente etapa y con qué evidencia?
```

Esa evidencia puede ser:

- Fundamental: métricas contables, valuación, rentabilidad, crecimiento,
  liquidez, leverage y cash flow.
- Estadística: menor correlación relativa dentro de grupos o contra un
  portafolio existente.

## Qué No Resuelve

`selection` no debe confundirse con:

- Optimización de pesos.
- Ejecución de órdenes.
- Backtesting completo.
- Valuación intrínseca completa.
- Recomendación financiera final.
- Validación profesional de datos de terceros.

La salida de `selection` es una lista corta y una explicación. La decisión de
capital ocurre después.

## Dos Rutas De Selección

### Ruta Fundamental

La ruta fundamental parte de tickers y datos financieros. Construye métricas,
aplica una configuración de scoring y produce rankings.

```text
tickers
  -> YahooFundamentalsProvider
  -> FundamentalData
  -> build_metrics_frame
  -> score_fundamentals
  -> FundamentalSelector.rank
  -> select_top / selection_report
```

Esta ruta es útil cuando se quiere representar una tesis de inversión, por
ejemplo `value` o `growth`.

### Ruta De Correlación

La ruta de correlación parte de grupos de tickers y precios históricos. Calcula
retornos, matrices de correlación y rankings por menor correlación promedio.

```text
grouped_tickers
  -> descarga de precios
  -> filtro de cobertura
  -> retornos
  -> matriz de correlación
  -> corr_score
  -> candidatos diversificados
```

Esta ruta es útil cuando la pregunta no es "qué empresa parece mejor", sino
"qué activo agrega menos redundancia estadística".

## Contrato Conceptual De Salida

El contrato natural de `selection` es:

```text
selected_tickers + evidencia de selección
```

En fundamental, la evidencia incluye `fundamental_score`, componentes de score,
métricas y `score_coverage`. En correlación, la evidencia incluye
`corr_score`, matrices de correlación, candidatos por grupo y selección final.

## Relación Con Portfolio Management

La secuencia conceptual del proyecto es:

```text
selection -> optimization -> portfolio construction -> backtesting -> risk/reporting
```

La documentación de CFA Institute sobre portfolio management enfatiza que la
construcción de portafolios requiere analizar características de riesgo y
retorno, correlaciones y preferencias del inversionista. `selection` representa
la etapa previa: definir un universo candidato antes de optimizar o evaluar
resultados.

## Fuentes Conceptuales

- CFA Institute, *Portfolio Risk and Return: Part I*. Usada para el papel de
  riesgo, retorno, correlación y diversificación en construcción de portafolios.
  Consultado: 2026-05-03.
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/portfolio-risk-return-part-1
- CFA Institute, *Active Equity Investing: Strategies*. Usada para distinguir
  selección fundamental, cuantitativa, value, growth y enfoques activos.
  Consultado: 2026-05-03.
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/active-equity-investing-strategies

