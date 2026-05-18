# Articulo Complementario: Regimen Esperado Y Vista De Activos

## Indice Del Documento

1. Mapa De La Serie
2. Problema Que Resuelve
3. Fundamento Conceptual
4. Fuentes Academicas O Profesionales
5. Traduccion Al Codigo
6. Clases, Metodos Y Atributos Implicados
7. Ejemplo Interpretado
8. Supuestos Y Limites
9. Errores Comunes
10. Relacion Con Workflows
11. Referencias

## Mapa De La Serie

| Orden | Documento | Rol |
|---:|---|---|
| 00 | `00_index.md` | Entrada secuencial. |
| 01 | `01_audit_snapshot.md` | Estado real. |
| 02 | `02_conceptual_model.md` | Modelo mental. |
| 03 | `03_module_boundary.md` | Frontera. |
| 04 | `04_architecture.md` | Arquitectura. |
| 05 | `05_capability_matrix.md` | Capacidades. |
| 06 | `06_contracts.md` | Contratos. |
| 07 | `07_workflows.md` | Workflows. |
| 08 | `08_public_api.md` | API publica. |
| 09 | `09_classes_and_methods.md` | Clases y metodos. |
| 10 | `10_examples.md` | Ejemplos. |
| 11 | `11_validation_and_edge_cases.md` | Validacion. |
| 12 | `12_operational_notes.md` | Operacion. |
| 13 | `13_glossary.md` | Glosario. |
| 15 | `15_expected_regime_and_asset_view_article.md` | Articulo complementario. |

## Problema Que Resuelve

El regimen actual es una fotografia. La seleccion de activos necesita tambien
una vision probable del futuro cercano. Este articulo explica como el modulo
representa esa vision y la convierte en una postura top-down de activos.

La salida intermedia es `AssetClassView`:

```text
CurrentRegimeSnapshot + RegimeForecast
-> current_stance + expected_stance + final_stance
-> overweights, underweights, preferred_styles, risk_controls
```

## Fundamento Conceptual

La logica forward-looking tiene dos principios:

- La postura de activos debe reflejar tanto condiciones actuales como escenario
  esperado.
- La liquidez puede funcionar como veto: si el regimen actual es `estresada`,
  el modelo conserva una postura defensiva aunque el forecast base sea mas
  favorable.

La idea no es predecir retornos exactos. Es ordenar la investigacion de manera
auditable: que escenario esperamos, con que probabilidad, por que, y que tipo
de exposicion parece tener mejor asimetria.

## Fuentes Academicas O Profesionales

La separacion entre liquidez de mercado y liquidez de fondeo esta respaldada por
Brunnermeier y Pedersen, quienes muestran que ambas pueden reforzarse y producir
espirales de liquidez. En gestion de portafolios, CFA Institute enfatiza que la
asignacion de activos traduce objetivos, restricciones y horizonte en
exposiciones, y que la asignacion tactica debe estar limitada por politica.

## Traduccion Al Codigo

`ForecastScenario` captura un escenario:

- `name`
- `macro_regime`
- `liquidity_regime`
- `probability`
- `horizon_months`
- `confidence`
- `rationale`
- `signals`

`RegimeForecast` agrupa escenarios y expone:

- `base_case`
- `expected_macro_regime`
- `expected_liquidity_regime`
- `horizon_months`
- `confidence`
- `scenario_table()`

`build_asset_class_view` hace la traduccion a postura:

1. Convierte regimen macro a bucket interno.
2. Convierte regimen de liquidez a bucket interno.
3. Busca la postura en `_STANCE_MATRIX`.
4. Decide si pesa mas el regimen actual o esperado.
5. Aplica veto si `liquidity_stress_veto=True` y la liquidez actual es
   `estresada`.
6. Devuelve detalles desde `_STANCE_DETAILS`.

## Clases, Metodos Y Atributos Implicados

| Objeto | Archivo | Papel |
|---|---|---|
| `ForecastScenario` | `forecast.py` | Escenario in-house puntual. |
| `RegimeForecast` | `forecast.py` | Conjunto de escenarios y base case. |
| `AssetClassView` | `asset_view.py` | Vista top-down accionable antes del cliente. |
| `build_asset_class_view` | `asset_view.py` | Funcion que cruza estado actual y futuro esperado. |
| `_STANCE_MATRIX` | `asset_view.py` | Matriz regimen-activos. |
| `_STANCE_DETAILS` | `asset_view.py` | Traduccion de stance a tilts y controles. |

La ubicacion de estas clases en el pipeline esta en
[macro_liquidity_module_architecture.drawio](./diagrams/macro_liquidity_module_architecture.drawio).

## Ejemplo Interpretado

Un forecast base puede decir:

```python
ForecastScenario(
    name="base",
    macro_regime="desaceleracion_ordenada",
    liquidity_regime="restrictiva",
    probability=0.65,
    confidence=0.70,
)
```

Si el regimen actual es expansion desinflacionaria con liquidez expansiva, pero
el forecast base apunta a desaceleracion con liquidez restrictiva, el modelo da
mas peso a la postura esperada cuando `expected_weight >= current_weight`. Esa
decision produce una vista mas selectiva, con controles de calidad, liquidez y
drawdown antes de entrar a selection.

## Supuestos Y Limites

- Las probabilidades del forecast son inputs de research, no estimaciones
  calibradas por el modulo.
- `base_case` es el escenario de mayor probabilidad; escenarios alternativos se
  reportan pero no ponderan directamente la matriz de activos.
- `_STANCE_MATRIX` es una matriz inicial de investigacion; debe calibrarse con
  evidencia historica y comite.
- Las etiquetas como `quality_equity` o `cash_like` son universos/estilos, no
  instrumentos especificos.
- El veto de liquidez es conservador y puede reducir sensibilidad al forecast.

## Errores Comunes

- Usar `RegimeForecast` como si fuera modelo estadistico automatico.
- Confundir `confidence` con probabilidad.
- Pasar probabilidades negativas o inconsistentes sin documentar racional.
- Saltar directo de `AssetClassView` a compra de instrumentos sin IPS.
- Ignorar escenarios downside cuando el contexto de liquidez se deteriora.

## Relacion Con Workflows

En `07_workflows.md`, el forecast se define despues de revisar `result.current`.
La vista de activos se construye antes de aplicar `ClientPolicy`, para preservar
la separacion entre research economico y restricciones del cliente.

El diagrama operativo esta en
[macro_liquidity_workflow.drawio](./diagrams/macro_liquidity_workflow.drawio).

## Referencias

- Markus K. Brunnermeier y Lasse Heje Pedersen, Market Liquidity and Funding
  Liquidity, NBER Working Paper 12939. URL:
  https://www.nber.org/papers/w12939. Consultado: 2026-05-08. Referencia para
  la relacion entre liquidez de mercado, fondeo y riesgo.
- CFA Institute, Principles of Asset Allocation. URL:
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/principles-asset-allocation.
  Consultado: 2026-05-08. Referencia para asset allocation como traduccion de
  objetivos, restricciones y exposiciones.
- CFA Institute, Asset Allocation with Real-World Constraints. URL:
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/asset-allocation-with-real-world-constraints.
  Consultado: 2026-05-08. Referencia para restricciones, liquidez y ajustes
  tacticos.
