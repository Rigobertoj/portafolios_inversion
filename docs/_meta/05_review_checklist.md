# Checklist De Revisión Documental

## Índice Del Documento

1. Propósito
2. Checklist De Estructura
3. Checklist De Contenido
4. Checklist De Fuentes
5. Checklist De Diagramas
6. Checklist De Código

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

Este checklist se usa antes de considerar terminada la documentación de un
módulo.

## Checklist De Estructura

- [ ] Existe `00_index.md`.
- [ ] El `README.md` indica qué leer primero.
- [ ] Cada documento tiene índice propio.
- [ ] Cada documento tiene mapa de la serie.
- [ ] Los nombres están numerados y ordenados.
- [ ] Los documentos viejos, si existen, están enlazados como referencia
      heredada o complementaria.

## Checklist De Contenido

- [ ] El modelo conceptual explica qué resuelve y qué no resuelve el módulo.
- [ ] La arquitectura mapea archivos a responsabilidades.
- [ ] Los contratos describen entradas, salidas y estados derivados.
- [ ] Los artículos conectan teoría, código e interpretación.
- [ ] Los workflows muestran usos reales.
- [ ] La API reference no sustituye los artículos.
- [ ] Los edge cases explican errores frecuentes y supuestos.

## Checklist De Fuentes

- [ ] Cada artículo tiene referencias.
- [ ] Las fuentes profesionales o académicas están separadas de las fuentes de
      implementación.
- [ ] Las fuentes web tienen URL y fecha de consulta cuando aplica.
- [ ] No hay citas largas innecesarias.

## Checklist De Diagramas

- [ ] Los diagramas existen o el documento explica que están pendientes.
- [ ] Los `.drawio` están enlazados.
- [ ] Las imágenes exportadas, si existen, están en `diagrams/exports/`.
- [ ] Los artículos enlazan el diagrama cerca del concepto explicado.

## Checklist De Código

- [ ] Las clases mencionadas existen.
- [ ] Los métodos mencionados existen.
- [ ] Los atributos mencionados existen.
- [ ] Los ejemplos usan imports públicos cuando sea posible.
- [ ] Las afirmaciones sobre estados internos se verificaron contra código.

