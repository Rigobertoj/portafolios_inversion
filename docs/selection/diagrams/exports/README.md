# Selection Diagram Exports

## Índice Del Documento

1. Propósito
2. Mapa De La Serie
3. Estado Actual
4. Regla De Exportación

## Mapa De La Serie

| Orden | Documento | Rol |
|---:|---|---|
| 00 | `../README.md` | Entrada de diagramas. |
| 01 | `../selection_module_architecture.drawio` | Fuente editable de arquitectura. |
| 02 | `../selection_class_architecture.drawio` | Fuente editable de clases. |
| 03 | `../selection_workflow.drawio` | Fuente editable de workflow. |
| 04 | `exports/README.md` | Estado de exportaciones a imagen. |

## Propósito

Esta carpeta reserva el lugar para imágenes exportadas de los diagramas de
`selection`. Los `.drawio` siguen siendo la fuente editable.

## Estado Actual

Las imágenes `.png` o `.svg` están pendientes de exportación porque el entorno
actual no tiene una herramienta CLI de diagrams.net disponible.

## Regla De Exportación

Cuando exista `drawio` o una herramienta equivalente:

```text
selection_module_architecture.drawio -> exports/selection_module_architecture.png
selection_class_architecture.drawio -> exports/selection_class_architecture.png
selection_workflow.drawio -> exports/selection_workflow.png
```

Después de exportar, los artículos conceptuales pueden incrustar esas imágenes
cerca del texto que explican.

