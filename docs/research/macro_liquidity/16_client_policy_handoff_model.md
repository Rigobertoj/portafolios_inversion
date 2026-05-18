# Articulo Complementario: Cliente Y Handoff A Selection

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Frontera Entre Research Y Cliente
4. Politica Del Cliente
5. Traduccion A `SelectionContext`
6. Impacto En Selection
7. Ejemplo De Handoff
8. Supuestos Y Limites
9. Referencias

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
| 16 | `16_client_policy_handoff_model.md` | Articulo complementario. |

## Proposito

Este documento explica como la postura macro/liquidez se adapta al cliente sin
contaminar el diagnostico economico. La idea central es:

```text
research economico reusable + politica especifica del cliente
-> SelectionContext especifico del cliente
```

## Frontera Entre Research Y Cliente

`macro_liquidity` no contiene moneda base, max drawdown ni exclusiones eticas.
Esos datos viven en `client_policy`. La primera funcion que une ambos mundos es
`build_selection_context`.

Esta frontera permite que el mismo regimen macro/liquidez pueda usarse para:

- Un cliente mexicano con base MXN.
- Un cliente estadounidense con base USD.
- Un mandato institucional con restricciones propias.
- Un research interno sin cliente final todavia.

## Politica Del Cliente

`ClientPolicy` se compone de:

| Objeto | Campos Principales |
|---|---|
| `ClientProfile` | `client_type`, `base_currency`, `horizon_years`, `risk_profile`, `max_drawdown`, `liquidity_need`, `contribution_style`. |
| `InvestmentMandate` | `target_return_range`, `eligible_assets`, `excluded_sectors`, `eligible_regions`, `benchmark_currency`, `notes`. |

La funcion `mexican_moderate_aggressive_growth_policy()` codifica el cliente
actual usado en el proyecto: persona mexicana, base MXN, horizonte de diez anos,
perfil moderado-agresivo, drawdown maximo de 20%, aportaciones periodicas y
exclusiones del IPS.

## Traduccion A `SelectionContext`

`build_selection_context` recibe:

- `CurrentRegimeSnapshot`
- `RegimeForecast`
- `AssetClassView`
- `ClientPolicy`

Y produce:

- Regimen actual y esperado.
- Moneda base y perfil de riesgo.
- Overweights y underweights de asset class.
- Estilos preferidos.
- Sectores excluidos.
- Activos elegibles.
- Controles de riesgo.
- `score_tilts`.
- Racional auditable.

El diagrama de clase muestra este cruce en la caja de `build_selection_context`:
[macro_liquidity_class_architecture.drawio](./diagrams/macro_liquidity_class_architecture.drawio).

## Impacto En Selection

`SelectionContext` puede afectar selection en cuatro planos:

| Plano | Campo |
|---|---|
| Universo | `eligible_assets`, `asset_class_overweights`, `asset_class_underweights`. |
| Filtros | `excluded_sectors`, `risk_controls`. |
| Scoring | `preferred_styles`, `score_tilts`. |
| Auditoria | regimenes y `rationale`. |

Ejemplo: si la vista final favorece calidad y liquidez, `score_tilts` puede
aumentar el peso de rentabilidad, flujo de efectivo, bajo apalancamiento y
liquidez en el scoring fundamental.

## Ejemplo De Handoff

```python
client_policy = mexican_moderate_aggressive_growth_policy()

selection_context = build_selection_context(
    current=result.current,
    forecast=forecast,
    asset_view=asset_view,
    policy=client_policy,
)
```

El resultado puede visualizarse en notebook con:

```python
selection_context.to_frame()
```

## Supuestos Y Limites

- `SelectionContext` no obliga a selection a implementar todos los tilts; define
  el contrato de contexto.
- Las exclusiones sectoriales dependen de que selection tenga metadatos
  suficientes para filtrarlas.
- `score_tilts` es una primera traduccion heuristica de estilos a factores de
  scoring.
- La politica de cliente actual es un helper especifico; para multiples
  clientes debe crearse una fabrica o carga desde configuracion externa.

## Referencias

- CFA Institute, Standard III(C) Suitability. URL:
  https://www.cfainstitute.org/standards/professionals/code-ethics-standards/standards-of-practice-iii-c.
  Consultado: 2026-05-08. Referencia para consistencia entre recomendacion,
  situacion del cliente, objetivos y restricciones.
- CFA Institute, Portfolio Management: An Overview. URL:
  https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/portfolio-management-overview.
  Consultado: 2026-05-08. Referencia para el orden cliente, asset allocation,
  security analysis y portfolio construction.
