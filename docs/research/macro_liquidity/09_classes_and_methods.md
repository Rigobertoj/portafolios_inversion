# Clases Y Metodos De `macro_liquidity`

## Indice Del Documento

1. Proposito
2. Mapa De La Serie
3. Catalogo
4. Providers
5. Transformaciones
6. Regimen Actual
7. Forecast
8. Vista De Activos
9. Cliente Y Strategy

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

Referencia tecnica enfocada en parametros, retornos, excepciones y relaciones
entre objetos.

## Catalogo

### `EconomicSeriesSpec`

```python
EconomicSeriesSpec(
    name: str,
    engine: Literal["macro", "liquidity"],
    block: str,
    higher_is_better: bool,
    provider: str = "",
    provider_code: str = "",
    frequency: str = "",
    transform: Literal["level", "diff", "pct_change", "yoy_change"] = "level",
    periods: int = 1,
    weight: float = 1.0,
    required: bool = False,
    description: str = "",
)
```

Uso: describir una serie antes de descargarla o scorearla.

### `default_us_macro_liquidity_catalog()`

Retorna el catalogo base. Contiene series FRED y una serie Banxico (`usd_mxn`).
Con `FredApiProvider`, la serie Banxico se ignora porque no es FRED.

## Providers

### `ProviderConfig(env=None, connection_specs=...)`

Metodos:

- `token_for(provider) -> str | None`
- `is_available(provider) -> bool`
- `availability_report() -> pd.DataFrame`
- `require_token(provider) -> str`

Excepcion: `require_token` levanta `MissingCredentialError` si falta token.

### `LocalSeriesProvider(frame)`

Constructor:

```python
LocalSeriesProvider(frame: pd.DataFrame)
```

Metodo:

```python
fetch(specs, start=None, end=None) -> pd.DataFrame
```

Uso: filtra un `DataFrame` local por los nombres de `specs` y rango de fechas.

### `FredApiProvider(config=None, timeout=30)`

Metodo:

```python
fetch(specs, start=None, end=None) -> pd.DataFrame
```

Uso: descarga solo specs con `provider.upper() == "FRED"` y `provider_code`.
Internamente llama `ProviderConfig.require_token("FRED")`.

## Transformaciones

### `normalize_series_frame(frame)`

Input: `DataFrame` con `date`, `series`, `value`.

Output: `DataFrame` normalizado y ordenado.

Excepcion: `ValueError` si faltan columnas.

### `build_indicator_panel(series_frame, specs, min_periods=6)`

Output: panel largo con metadata, `signal`, `zscore`, `signed_zscore`.

### `build_score_frame(indicators)`

Output: tabla por fecha con scores de bloque, `macro_score`,
`liquidity_score`, `macro_coverage`, `liquidity_coverage`.

## Regimen Actual

### `MacroLiquidityResearch(series_specs=None, min_periods=6)`

Guarda:

- `series_specs`
- `min_periods`, forzado a minimo 2

Metodo:

```python
analyze_current(series_frame: pd.DataFrame) -> MacroLiquidityResult
```

Output:

- `indicators`
- `scores`
- `current`

Excepcion: `ValueError` si los scores quedan vacios.

## Forecast

### `ForecastScenario`

Campos: `name`, `macro_regime`, `liquidity_regime`, `probability`,
`horizon_months`, `confidence`, `rationale`, `signals`.

Metodo:

```python
normalized_probability(total: float) -> float
```

### `RegimeForecast`

Campos: `scenarios`, `method`, `as_of`.

Propiedades:

- `base_case`
- `expected_macro_regime`
- `expected_liquidity_regime`
- `horizon_months`
- `confidence`

Metodo:

```python
scenario_table() -> pd.DataFrame
```

Excepcion: `ValueError` si `scenarios` esta vacio.

## Vista De Activos

### `build_asset_class_view`

```python
build_asset_class_view(
    current: CurrentRegimeSnapshot,
    forecast: RegimeForecast,
    current_weight: float = 0.40,
    expected_weight: float = 0.60,
    liquidity_stress_veto: bool = True,
) -> AssetClassView
```

Regla: si `liquidity_stress_veto=True` y la liquidez actual es `estresada`, la
postura final usa `current_stance`. Si no, usa la postura esperada cuando
`expected_weight >= current_weight`.

### `AssetClassView.to_frame()`

Retorna una tabla compacta para notebooks.

## Cliente Y Strategy

### `mexican_moderate_aggressive_growth_policy()`

Retorna `ClientPolicy` del cliente actual.

### `build_selection_context`

```python
build_selection_context(
    current: CurrentRegimeSnapshot,
    forecast: RegimeForecast,
    asset_view: AssetClassView,
    policy: ClientPolicy,
) -> SelectionContext
```

Agrega controles de cliente y calcula `score_tilts`.

### `SelectionContext`

Metodos:

- `as_dict() -> dict[str, object]`
- `to_frame() -> pd.DataFrame`
