# API Publica De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Imports Recomendados
4. API De Research Macro/Liquidez
5. API De Cliente
6. API De Strategy
7. Que No Es API Publica Estable

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

## Proposito

Este documento dice que imports usar desde notebooks o modulos externos. El
detalle de parametros y retornos esta en `09_classes_and_methods.md`.

## Imports Recomendados

```python
from src.research.macro_liquidity import (
    ForecastScenario,
    FredApiProvider,
    LocalSeriesProvider,
    MacroLiquidityResearch,
    ProviderConfig,
    RegimeForecast,
    build_asset_class_view,
    default_us_macro_liquidity_catalog,
)
from src.client_policy import mexican_moderate_aggressive_growth_policy
from src.strategy import build_selection_context
```

## API De Research Macro/Liquidez

Exportado desde `src.research.macro_liquidity`:

| Nombre | Uso |
|---|---|
| `EconomicSeriesSpec` | Definir metadata de una serie. |
| `default_us_macro_liquidity_catalog` | Obtener catalogo base. |
| `ProviderConfig` | Revisar credenciales y disponibilidad. |
| `LocalSeriesProvider` | Usar datos locales. |
| `FredApiProvider` | Descargar FRED. |
| `MissingCredentialError` | Manejar falta de credenciales. |
| `MacroLiquidityResearch` | Ejecutar diagnostico actual. |
| `MacroLiquidityResult` | Bundle de resultado actual. |
| `CurrentRegimeSnapshot` | Snapshot actual. |
| `ForecastScenario` | Escenario esperado. |
| `RegimeForecast` | Forecast in-house. |
| `AssetClassView` | Vista top-down. |
| `build_asset_class_view` | Construir vista top-down. |
| `classify_macro_regime` | Clasificar macro desde fila de scores. |
| `classify_liquidity_regime` | Clasificar liquidez desde score. |

## API De Cliente

Exportado desde `src.client_policy`:

| Nombre | Uso |
|---|---|
| `ClientProfile` | Caracteristicas del cliente. |
| `InvestmentMandate` | Universo, exclusiones y mandato. |
| `ClientPolicy` | Perfil + mandato. |
| `mexican_moderate_aggressive_growth_policy` | Helper del cliente actual. |

## API De Strategy

Exportado desde `src.strategy`:

| Nombre | Uso |
|---|---|
| `SelectionContext` | Contrato final hacia selection. |
| `build_selection_context` | Construir `SelectionContext`. |

## Que No Es API Publica Estable

Los nombres privados con `_`, como `_STANCE_MATRIX`, `_score_tilts` o
`_transform_values`, son detalles internos. Pueden documentarse para entender el
comportamiento, pero no conviene depender de ellos desde notebooks o modulos
externos.
