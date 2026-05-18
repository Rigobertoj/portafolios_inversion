# Diagramas De `macro_liquidity`

## Indice Del Documento

1. Mapa De La Serie
2. Lectura
3. Archivos
4. Enlaces Directos
5. Exportaciones A Imagen

## Mapa De La Serie

| Orden | Documento | Rol |
|---:|---|---|
| 00 | [../00_index.md](../00_index.md) | Entrada secuencial. |
| 01 | [../01_audit_snapshot.md](../01_audit_snapshot.md) | Estado real del modulo. |
| 02 | [../02_conceptual_model.md](../02_conceptual_model.md) | Concepto y frontera del modulo. |
| 04 | [../04_architecture.md](../04_architecture.md) | Arquitectura y diagrama de modulo. |
| 06 | [../06_contracts.md](../06_contracts.md) | Contratos y diagrama de clases. |
| 07 | [../07_workflows.md](../07_workflows.md) | Workflow y diagrama operativo. |
| 08 | [../08_public_api.md](../08_public_api.md) | API publica. |
| 09 | [../09_classes_and_methods.md](../09_classes_and_methods.md) | API y relaciones de clases. |

## Lectura

Estos diagramas documentan el sistema nuevo que conecta investigacion
macroeconomica, liquidez, politica del cliente y handoff hacia `selection`.

El limite principal es deliberado:

```text
research macro/liquidity -> client policy overlay -> SelectionContext -> selection
```

`macro_liquidity` diagnostica regimenes y produce una vista top-down. No escoge
tickers. `client_policy` contiene el IPS y no pronostica la economia.
`strategy` traduce ambos mundos a un contrato que `selection` puede consumir.

## Archivos

1. [macro_liquidity_module_architecture.drawio](./macro_liquidity_module_architecture.drawio)  
   Vista universo: fuentes, catalogo, transformaciones, regimen actual,
   forecast, vista de activos, politica de cliente y handoff.

2. [macro_liquidity_class_architecture.drawio](./macro_liquidity_class_architecture.drawio)  
   Vista de clases, funciones y contratos: `ProviderConfig`,
   `EconomicSeriesSpec`, `MacroLiquidityResearch`, `RegimeForecast`,
   `AssetClassView`, `ClientPolicy` y `SelectionContext`.

3. [macro_liquidity_workflow.drawio](./macro_liquidity_workflow.drawio)  
   Vista operacional para notebook: definir policy/specs, validar providers,
   construir features, correr nowcast, seleccionar modelos de series de tiempo,
   clasificar regimen, aplicar escenarios y producir `SelectionContext`.

4. [../18_policy_workflow_diagram.md](../18_policy_workflow_diagram.md)  
   Documento Markdown con diagramas Mermaid, tabla de superficie modificable y
   lectura operativa de `MacroLiquidityPolicy`.

## Enlaces Directos

Para abrir copias editables en diagrams.net:

- [open_in_diagrams_net.md](./open_in_diagrams_net.md)

Los archivos `.drawio` tambien pueden abrirse con `File -> Open From -> Device`
en diagrams.net.

## Exportaciones A Imagen

La carpeta [exports](./exports/README.md) reserva el lugar para imagenes `.png`
o `.svg`. Por ahora los artefactos canonicos son los `.drawio` y los enlaces
directos a diagrams.net.
