# Diseño De Implementación Del Módulo De Reporteo

> Estado: borrador de diseño temporal.  
> Ubicación propuesta: `docs/reporting/`, carpeta destinada a concentrar la
> documentación futura del módulo de reporteo.

## 1. Propósito Del Módulo

El módulo de reporteo debe convertirse en la capa narrativa y estructural que
traduce resultados cuantitativos en entregables interpretables. Su objetivo no
es calcular por separado cada métrica financiera, ni reemplazar a `portfolio`,
`backtesting` o `risk`. Su responsabilidad es tomar los artefactos que esos
módulos ya producen, ordenarlos bajo una lógica común y devolver tablas,
secciones, diagnósticos y objetos listos para análisis, notebooks o documentos.

En términos del flujo completo del proyecto:

```text
selection
    selecciona candidatos

optimization
    calcula pesos bajo un objetivo

backtesting
    simula desempeño estatico o dinamico

portfolio / risk
    calculan metricas de desempeno y exposicion

reporting
    organiza, compara, explica y entrega resultados
```

La intención es que `reporting` sea el punto donde el proyecto deja de mostrar
salidas sueltas y empieza a presentar una historia coherente:

- qué universo se eligió,
- qué estrategia se optimizó,
- cómo se comportó dentro y fuera de muestra,
- qué riesgos aparecieron,
- qué costos o rebalanceos afectaron el resultado,
- y qué decisión práctica se desprende del análisis.

## 2. Problema Que Resuelve

Actualmente el repositorio ya tiene piezas de reporteo distribuidas:

- `PortfolioPerformanceAnalysis` ofrece una interfaz orientada a reporte para
  métricas de desempeño de un `Portfolio`.
- `PerformanceMetricsCalculator` centraliza métricas compartidas entre
  portafolios y backtests.
- `RiskAnalyzer` agrega drawdown, VaR/CVaR, volatilidad, tracking error e
  information ratio.
- `BacktestResult` y `DynamicBacktestResult` almacenan retornos, evolución,
  métricas netas, métricas brutas, métricas pre-backtest, costos, turnover y
  pesos históricos.

El problema es que esas piezas todavía no viven bajo un contrato narrativo
único. Cada módulo produce buenos datos, pero el usuario necesita decidir cómo
compararlos, cómo nombrarlos, qué poner primero y qué separar en secciones.

El módulo de reporteo debe resolver esa dispersión mediante una capa que:

1. Normalice entradas de distintos módulos.
2. Preserve tablas técnicas sin perder contexto.
3. Genere secciones ordenadas de reporte.
4. Distinga métricas in-sample, out-of-sample, gross, net y execution.
5. Permita comparar estrategias de forma consistente.
6. Deje trazabilidad entre cada sección y su fuente de datos.

## 3. Principio Arquitectónico

La idea central es separar tres responsabilidades:

```text
calculo financiero -> agregacion analitica -> presentacion narrativa
```

El cálculo financiero debe seguir en módulos especializados:

- `portfolio.metrics_basic`
- `portfolio.metrics_downside`
- `portfolio.benchmark_analysis`
- `portfolio.performance_metrics`
- `risk.drawdown`
- `risk.var_cvar`
- `risk.volatility`
- `risk.tracking`
- `backtesting.engine_static`
- `backtesting.engine_dynamic`

La agregación analítica debe ordenar esos cálculos en objetos intermedios:

- resumen de performance,
- resumen de riesgo,
- resumen de backtesting,
- resumen de ejecución,
- comparación entre estrategias,
- diagnóstico de consistencia.

La presentación narrativa debe transformar los objetos intermedios en una
estructura legible:

- títulos de secciones,
- tablas,
- notas metodológicas,
- advertencias,
- conclusiones,
- anexos.

El módulo de reporteo no debe duplicar fórmulas. Debe usar calculadoras y
analizadores existentes. Si una métrica falta, primero debe agregarse en el
módulo especializado correspondiente y luego exponerse en el reporte.

## 4. Alcance Inicial

La primera versión del módulo debe cubrir tres estilos de reporte.

### 4.1 Reporte De Portafolio

Entrada principal:

```python
Portfolio
```

Entradas opcionales:

```python
benchmark_returns
benchmark_prices
risk_free_rate
threshold
initial_value
trading_days
```

Fuentes internas:

- `PortfolioPerformanceAnalysis`
- `RiskAnalyzer`

Secciones esperadas:

1. Resumen ejecutivo.
2. Composición del portafolio.
3. Métricas de desempeño.
4. Métricas downside.
5. Métricas relativas a benchmark.
6. Riesgo histórico.
7. Drawdown.
8. Lectura metodológica.

### 4.2 Reporte De Backtesting Estático

Entrada principal:

```python
BacktestResult
```

Fuentes internas:

- `result.pre_back_metrics`
- `result.gross_metrics`
- `result.net_metrics`
- `result.execution_metrics`
- `result.metrics`
- `result.returns`
- `result.evolution`
- `result.strategy_results`

Secciones esperadas:

1. Configuración de la simulación.
2. Estrategias evaluadas.
3. Pesos optimizados por estrategia.
4. Métricas in-sample.
5. Métricas out-of-sample gross.
6. Métricas out-of-sample net.
7. Comparación contra benchmark.
8. Evolución del capital.
9. Diagnóstico final.

### 4.3 Reporte De Backtesting Dinámico

Entrada principal:

```python
DynamicBacktestResult
```

Fuentes internas:

- `result.pre_back_metrics`
- `result.gross_metrics`
- `result.net_metrics`
- `result.execution_metrics`
- `result.weights_history`
- `result.turnover`
- `result.transaction_costs`
- `result.strategy_results`

Secciones esperadas:

1. Configuración general.
2. Política de rebalanceo.
3. Historial de pesos.
4. Turnover por fecha y estrategia.
5. Costos de transacción.
6. Métricas pre-back promedio.
7. Desempeño gross vs net.
8. Evolución de capital.
9. Diagnóstico de estabilidad.
10. Conclusión de implementación.

## 5. Diseño De Carpetas

La carpeta documental futura puede quedar así:

```text
docs/reporting/
├── README.md
├── architecture.md
├── api_reference.md
├── workflows.md
├── implementation_design.md
└── diagrams/
    ├── README.md
    ├── reporting_module_architecture.drawio
    ├── reporting_class_architecture.drawio
    └── reporting_workflow.drawio
```

La carpeta de código sugerida sería:

```text
src/reporting/
├── __init__.py
├── configs.py
├── sections.py
├── portfolio_report.py
├── backtest_report.py
├── risk_report.py
├── renderers.py
└── results.py
```

Esta separación permite implementar primero objetos de reporte en memoria y
dejar para después renderizadores más específicos como Markdown, HTML, Excel o
PDF.

## 6. Modelo Conceptual

El módulo debe trabajar con una abstracción central: un reporte no es solo una
tabla, sino una colección ordenada de secciones.

```text
Report
    ├── metadata
    ├── sections
    ├── tables
    ├── notes
    ├── warnings
    └── artifacts
```

Cada sección debe tener:

- identificador estable,
- título legible,
- descripción corta,
- fuente de datos,
- contenido tabular o textual,
- severidad opcional para advertencias,
- orden dentro del documento.

La ventaja de este diseño es que una misma estructura puede renderizarse de
distintas formas. Un notebook puede mostrar tablas de `pandas`; un documento
Markdown puede generar narrativa; un dashboard puede usar los mismos objetos
para tarjetas y gráficos.

## 7. Objetos Propuestos

### 7.1 `ReportConfig`

Configuración general del reporte.

Campos propuestos:

```python
@dataclass(frozen=True)
class ReportConfig:
    title: str
    subtitle: str | None = None
    initial_value: float = 1.0
    risk_free_rate: float = 0.0
    threshold: float = 0.0
    confidence_level: float = 0.95
    trading_days: int = 252
    include_methodology: bool = True
    include_warnings: bool = True
```

Responsabilidad:

- Centralizar supuestos de reporteo.
- Evitar pasar parámetros sueltos entre secciones.
- Alinear métricas que dependen de supuestos compartidos.

Validaciones:

- `initial_value > 0`.
- `0 < confidence_level < 1`.
- `trading_days > 0`.
- `title` no vacío.

### 7.2 `ReportSection`

Unidad mínima del reporte.

Campos propuestos:

```python
@dataclass
class ReportSection:
    key: str
    title: str
    body: str | None = None
    table: pd.DataFrame | pd.Series | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str | None = None
    order: int = 0
```

Responsabilidad:

- Representar un bloque lógico del reporte.
- Mantener trazabilidad hacia el módulo que lo alimenta.
- Permitir ordenamiento estable.

Ejemplos de `key`:

- `executive_summary`
- `portfolio_weights`
- `performance_metrics`
- `risk_summary`
- `pre_back_metrics`
- `gross_metrics`
- `net_metrics`
- `execution_metrics`
- `rebalance_history`

### 7.3 `ReportDocument`

Contenedor completo.

Campos propuestos:

```python
@dataclass
class ReportDocument:
    title: str
    sections: list[ReportSection]
    metadata: dict[str, object] = field(default_factory=dict)
```

Métodos propuestos:

```python
section(key: str) -> ReportSection
tables() -> dict[str, pd.DataFrame]
warnings() -> list[str]
to_markdown() -> str
```

Responsabilidad:

- Mantener el orden narrativo.
- Permitir acceso programático a secciones.
- Preparar el handoff hacia renderizadores.

### 7.4 `PortfolioReportBuilder`

Constructor de reportes para un `Portfolio`.

Firma propuesta:

```python
class PortfolioReportBuilder:
    def __init__(
        self,
        portfolio: Portfolio,
        config: ReportConfig | None = None,
        benchmark_returns: pd.Series | pd.DataFrame | None = None,
        benchmark_prices: pd.Series | pd.DataFrame | None = None,
        benchmark_name: str | None = None,
    ) -> None:
        ...

    def build(self) -> ReportDocument:
        ...
```

Responsabilidad:

- Leer pesos y retornos del `Portfolio`.
- Delegar métricas a `PortfolioPerformanceAnalysis`.
- Delegar riesgo a `RiskAnalyzer`.
- Construir secciones de composición, performance y riesgo.

### 7.5 `BacktestReportBuilder`

Constructor para `BacktestResult` y `DynamicBacktestResult`.

Firma propuesta:

```python
class BacktestReportBuilder:
    def __init__(
        self,
        result: BacktestResult | DynamicBacktestResult,
        config: ReportConfig | None = None,
    ) -> None:
        ...

    def build(self) -> ReportDocument:
        ...
```

Responsabilidad:

- Detectar si el resultado es estático o dinámico.
- Ordenar métricas por etapa: pre-back, gross, net y execution.
- Construir secciones de pesos finales o pesos históricos.
- Mostrar turnover y costos cuando existan.
- Generar advertencias si faltan benchmarks, costos o datos suficientes.

### 7.6 `MarkdownRenderer`

Renderizador inicial.

Firma propuesta:

```python
class MarkdownRenderer:
    def render(self, document: ReportDocument) -> str:
        ...
```

Responsabilidad:

- Convertir `ReportDocument` en Markdown.
- Mantener tablas de `pandas` en formato legible.
- Incluir notas y warnings por sección.

Debe ser el primer renderizador porque es simple, versionable y funciona bien
con notebooks, GitHub y la documentación del repositorio.

## 8. Contratos De Entrada

### 8.1 Contrato De `Portfolio`

El reporte de portafolio espera:

- `portfolio.tickers`
- `portfolio.weights`
- `portfolio.asset_returns()`
- `portfolio.portfolio_returns()`
- `portfolio.wealth_index(initial_value)`

No debe reconstruir retornos manualmente si `Portfolio` ya los ofrece.

### 8.2 Contrato De `BacktestResult`

El reporte estático espera:

- `config`
- `strategy_results`
- `returns`
- `evolution`
- `pre_back_metrics`
- `gross_metrics`
- `net_metrics`
- `execution_metrics`
- `metrics`

La sección de pesos debe leer:

```python
result.strategy_results[name].weights_by_ticker
```

### 8.3 Contrato De `DynamicBacktestResult`

El reporte dinámico espera:

- `config`
- `rebalance_config`
- `strategy_results`
- `weights_history`
- `turnover`
- `transaction_costs`
- `pre_back_metrics`
- `gross_metrics`
- `net_metrics`
- `execution_metrics`
- `metrics`

La sección de pesos debe priorizar `weights_history`, no solo el último vector.

## 9. Estructura Narrativa Del Reporte

El orden debe ser consistente entre estilos de reporte. Una propuesta base:

```text
1. Resumen ejecutivo
2. Contexto y supuestos
3. Construccion o estrategia evaluada
4. Desempeno
5. Riesgo
6. Benchmark y metricas relativas
7. Ejecucion, rebalanceo y costos
8. Alertas metodologicas
9. Conclusion
10. Anexos tecnicos
```

No todos los reportes tienen todas las secciones. Por ejemplo, un reporte de
portafolio simple puede no tener rebalanceo ni costos. Un backtest dinámico sí
debe tenerlos.

## 10. Métricas Base

El módulo debe respetar las etiquetas ya definidas en
`PerformanceMetricsCalculator`:

```text
Rendimiento esperado
Rendimiento realizado
Volatilidad
Ratio de sharpe
Downside risk
Upside risk
Omega
Beta
Alpha de Jensen
Ratio de Treynor
Ratio de Sortino
```

Estas etiquetas son importantes porque permiten comparar reportes de
portafolio, backtesting estático y backtesting dinámico sin cambiar nombres.

Métricas adicionales de ejecución:

```text
Turnover promedio
Turnover acumulado
Costos de transacción
Impacto de costos
```

Métricas adicionales de riesgo:

```text
Max Drawdown
VaR
CVaR
Tracking Error
Information Ratio
EWMA volatility
```

## 11. Advertencias Y Diagnósticos

El módulo debe producir warnings explícitos cuando una sección está incompleta.
Ejemplos:

- No hay benchmark configurado; beta, Jensen alpha y Treynor quedan como `NaN`.
- El benchmark no tiene suficiente intersección de fechas.
- El downside benchmark no tiene al menos dos observaciones alineadas.
- Hay estrategias sin costos de transacción reportados.
- El turnover es cero en todos los rebalanceos.
- El resultado incluye columnas con todos los valores `NaN`.
- El reporte dinámico tiene una sola fecha de rebalanceo, por lo que el
  diagnóstico de estabilidad es limitado.

Estas advertencias deben ser informativas, no excepciones, salvo que el reporte
no pueda construirse.

## 12. Reglas De Mutabilidad

El módulo de reporteo debe ser de solo lectura respecto a los resultados que
recibe.

No debe:

- modificar pesos del `Portfolio`,
- recalcular optimizaciones,
- reejecutar backtests,
- descargar precios,
- alterar `BacktestResult` o `DynamicBacktestResult`,
- rellenar datos faltantes en las fuentes originales.

Sí puede:

- copiar tablas,
- renombrar columnas de salida,
- ordenar secciones,
- construir tablas derivadas,
- crear diagnósticos,
- renderizar documentos.

## 13. Flujo De Implementación

### Fase 1. Estructuras Base

Crear:

- `src/reporting/__init__.py`
- `src/reporting/configs.py`
- `src/reporting/sections.py`
- `src/reporting/results.py`

Entregables:

- `ReportConfig`
- `ReportSection`
- `ReportDocument`
- tests unitarios de validación y orden de secciones.

### Fase 2. Reporte De Portafolio

Crear:

- `src/reporting/portfolio_report.py`

Implementar:

- `PortfolioReportBuilder`
- sección de pesos,
- sección de métricas de desempeño,
- sección de riesgo agregado,
- sección de benchmark cuando exista.

Pruebas:

- portafolio sin benchmark,
- portafolio con benchmark,
- portafolio con benchmark inválido,
- validación de warnings.

### Fase 3. Reporte De Backtesting

Crear:

- `src/reporting/backtest_report.py`

Implementar:

- soporte para `BacktestResult`,
- soporte para `DynamicBacktestResult`,
- detección de tipo de resultado,
- secciones pre-back, gross, net, execution,
- pesos estáticos,
- historial de pesos dinámicos,
- turnover y costos.

Pruebas:

- resultado estático con una estrategia,
- resultado estático con benchmark,
- resultado dinámico con rebalanceo mensual,
- comparación gross vs net,
- reporte con costos de transacción.

### Fase 4. Render Markdown

Crear:

- `src/reporting/renderers.py`

Implementar:

- `MarkdownRenderer`,
- `ReportDocument.to_markdown()`,
- salida estable para snapshots de tests.

Pruebas:

- render de secciones sin tabla,
- render de secciones con tabla,
- render de warnings,
- orden estable.

### Fase 5. Documentación Formal

Crear:

- `docs/reporting/README.md`
- `docs/reporting/architecture.md`
- `docs/reporting/api_reference.md`
- `docs/reporting/workflows.md`
- diagramas `.drawio`.

La documentación debe explicar tanto la arquitectura como el estilo narrativo:
por qué una métrica aparece en determinada sección y cómo interpretar gross,
net, pre-back y execution.

## 14. API Pública Esperada

La API mínima podría quedar así:

```python
from src.reporting import (
    BacktestReportBuilder,
    MarkdownRenderer,
    PortfolioReportBuilder,
    ReportConfig,
    ReportDocument,
    ReportSection,
)
```

Uso con portafolio:

```python
config = ReportConfig(
    title="Reporte de portafolio optimizado",
    initial_value=1_000_000,
    risk_free_rate=0.04,
)

report = PortfolioReportBuilder(
    portfolio=portfolio,
    config=config,
    benchmark_prices=benchmark_prices,
).build()

markdown = MarkdownRenderer().render(report)
```

Uso con backtesting:

```python
config = ReportConfig(
    title="Reporte de backtesting dinamico",
    initial_value=1_000_000,
    risk_free_rate=0.04,
)

report = BacktestReportBuilder(
    result=dynamic_result,
    config=config,
).build()

report.section("net_metrics").table
```

## 15. Criterios De Aceptación

La implementación inicial se considera completa si:

1. Existe una API pública `src.reporting`.
2. Puede construirse un reporte desde `Portfolio`.
3. Puede construirse un reporte desde `BacktestResult`.
4. Puede construirse un reporte desde `DynamicBacktestResult`.
5. Las secciones salen en orden estable.
6. Las tablas se copian sin mutar los objetos originales.
7. Las métricas usan `PerformanceMetricsCalculator`, `PortfolioPerformanceAnalysis`
   y `RiskAnalyzer` en vez de duplicar fórmulas.
8. Los warnings explican ausencia de benchmark o datos insuficientes.
9. Existe render Markdown.
10. Hay pruebas unitarias para contratos, warnings y render.

## 16. Riesgos De Diseño

### Riesgo 1. Duplicar Lógica Financiera

Si `reporting` empieza a recalcular Sharpe, beta, VaR o drawdown por su cuenta,
el proyecto tendrá resultados inconsistentes. La regla debe ser clara:
`reporting` orquesta, no formula.

### Riesgo 2. Mezclar Reporte Con Simulación

El reporte no debe ejecutar backtesting. Recibe un resultado ya calculado. Si se
mezclan ambas responsabilidades, será más difícil probar y explicar errores.

### Riesgo 3. Hacer El Renderizador Demasiado Temprano

Conviene construir primero `ReportDocument` y luego renderizar. Si se empieza
por Markdown directo, se pierde una representación programática reutilizable.

### Riesgo 4. Confundir Gross Y Net

En backtesting dinámico, las métricas netas deben reflejar costos de
transacción. El reporte debe nombrar explícitamente qué tabla viene de retornos
brutos y cuál viene de evolución neta.

### Riesgo 5. Ocultar Datos Faltantes

Los `NaN` no deben desaparecer sin explicación. Si una métrica relativa no puede
calcularse por falta de benchmark, el reporte debe mostrarlo como advertencia.

## 17. Relación Con Módulos Existentes

```text
src.portfolio.performance_metrics
    fuente comun de metricas de desempeno

src.portfolio.performance_analysis
    fachada para reportes de Portfolio

src.risk.report
    fachada para resumen de riesgo

src.backtesting.results
    contenedores de resultados estaticos y dinamicos

src.reporting
    capa nueva de organizacion, narrativa y render
```

La relación esperada es de dependencia hacia abajo:

```text
reporting -> portfolio / risk / backtesting
```

Los módulos inferiores no deben depender de `reporting`.

## 18. Resultado Esperado Para El Usuario

El usuario debería poder pasar de esto:

```python
result = DynamicBacktester(config, rebalance).run(strategies, prices=prices)
```

a esto:

```python
report = BacktestReportBuilder(result).build()
report.to_markdown()
```

sin decidir manualmente qué tablas juntar, en qué orden ponerlas o cómo explicar
la diferencia entre pre-back, gross, net y execution.

El valor del módulo está en convertir resultados técnicos correctos en una
lectura ordenada, comparable y defendible.
