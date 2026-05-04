# Política De Diagramas

## Índice Del Documento

1. Propósito
2. Cuándo Crear Diagramas
3. Tres Vistas Recomendadas
4. Exportación A Imágenes
5. Reglas De Inserción En Artículos
6. Validación

## Mapa De La Serie

| Orden | Documento | Rol |
|---:|---|---|
| 00 | `00_documentation_standard.md` | Estándar general. |
| 01 | `01_module_template.md` | Plantilla de módulo. |
| 02 | `02_article_template.md` | Plantilla de artículo. |
| 03 | `03_source_policy.md` | Política de fuentes. |
| 04 | `04_diagram_policy.md` | Política visual. |
| 05 | `05_review_checklist.md` | Checklist final. |

## Propósito

Los diagramas deben ayudar a entender arquitectura, relaciones y flujos. No son
decoración. En un módulo financiero, un buen diagrama debe mostrar qué datos se
mueven, qué decisión se toma y qué contrato recibe la siguiente etapa.

## Cuándo Crear Diagramas

Crear o actualizar diagramas cuando:

- Hay bifurcaciones metodológicas.
- Hay más de una clase coordinando una salida.
- Hay estados derivados o cachés.
- Hay transformaciones de datos encadenadas.
- Hay una diferencia importante entre concepto y método implementado.

## Tres Vistas Recomendadas

Cada módulo complejo debe tener:

| Diagrama | Propósito |
|---|---|
| `<module>_module_architecture.drawio` | Vista universo: módulos, fuentes, rutas y handoff. |
| `<module>_class_architecture.drawio` | Vista de clases, funciones, atributos y relaciones. |
| `<module>_workflow.drawio` | Vista práctica de uso de inicio a fin. |

## Exportación A Imágenes

Los `.drawio` son la fuente editable. Las imágenes exportadas viven en:

```text
docs/<module>/diagrams/exports/
```

Formatos recomendados:

- `.png` para documentación Markdown y sitios estáticos.
- `.svg` si el entorno lo renderiza bien y el texto debe escalar.

## Reglas De Inserción En Artículos

Un artículo debe incrustar un diagrama cuando el lector necesite ver:

- Una bifurcación de rutas.
- Un contrato de datos.
- Una relación clase-método-salida.
- Un workflow con decisiones.

Si el diagrama no aparece cerca del texto que explica, pierde utilidad.

## Validación

Antes de cerrar:

- Los `.drawio` deben abrir en diagrams.net.
- Los enlaces directos deben existir.
- Las imágenes exportadas deben corresponder al `.drawio` fuente.
- La documentación debe decir si una imagen está pendiente de exportación.

