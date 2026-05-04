# Estándar De Documentación Técnica Y Conceptual

## Índice Del Documento

1. Propósito
2. Principios
3. Estructura Reutilizable Por Módulo
4. Capas De Evidencia
5. Reglas De Navegación
6. Definición De Hecho
7. Criterios De Calidad

## Mapa De La Serie

| Orden | Documento | Rol |
|---:|---|---|
| 00 | `00_documentation_standard.md` | Define el sistema documental reutilizable. |
| 01 | `01_module_template.md` | Plantilla para documentar cualquier módulo. |
| 02 | `02_article_template.md` | Plantilla para artículos explicativos. |
| 03 | `03_source_policy.md` | Política de fuentes académicas, profesionales y técnicas. |
| 04 | `04_diagram_policy.md` | Política de diagramas, imágenes y artefactos visuales. |
| 05 | `05_review_checklist.md` | Checklist de revisión antes de cerrar documentación. |

## Propósito

Este estándar convierte la documentación del proyecto en un sistema escalable.
No está diseñado solo para `portafolios`; debe poder aplicarse a cualquier
repositorio futuro que tenga código, métodos, modelos, datos, decisiones de
diseño y conceptos que explicar.

La documentación debe resolver dos problemas al mismo tiempo:

- Ayudar a una persona a usar el código correctamente.
- Ayudar a una persona a entender por qué el código está estructurado así.

Por eso cada módulo combina explicación conceptual, arquitectura, contratos de
datos, workflows, referencia API, validaciones, diagramas y fuentes.

## Principios

1. La lectura debe tener un inicio evidente.
2. Cada documento debe tener índice propio.
3. Cada módulo debe tener una ruta numerada.
4. La API no sustituye la explicación conceptual.
5. El artículo explicativo es la pieza central cuando el módulo implementa un
   método, una teoría o una decisión analítica.
6. Las afirmaciones sobre el código se verifican contra archivos reales.
7. Las afirmaciones conceptuales se respaldan con fuentes académicas,
   profesionales u oficiales.
8. Los diagramas deben explicar relaciones y flujos, no solo decorar.

## Estructura Reutilizable Por Módulo

Cada módulo documentado debe seguir esta forma base:

```text
docs/<module>/
├── README.md
├── 00_index.md
├── 01_conceptual_model.md
├── 02_architecture.md
├── 03_data_contracts.md
├── 04_<main_method_or_theory>_article.md
├── 05_<secondary_method_or_theory>_article.md
├── 06_implementation_notes.md
├── 07_workflows.md
├── 08_api_reference.md
├── 09_validation_and_edge_cases.md
├── 10_glossary.md
└── diagrams/
```

La estructura puede crecer cuando el módulo lo justifique, pero no debe perder
el orden secuencial. Si un documento adicional es necesario, se numera y se
agrega al índice del módulo.

## Capas De Evidencia

La documentación distingue cinco capas:

| Capa | Fuente | Uso |
|---|---|---|
| A | Código, tests y notebooks del proyecto | Verdad sobre lo implementado. |
| B | Papers, libros o publicaciones académicas | Fundamento metodológico. |
| C | CFA Institute, GIPS u organismos profesionales | Interpretación financiera estándar. |
| D | Documentación oficial de librerías | Detalles de implementación. |
| E | Blogs, notas, cursos o artículos secundarios | Contexto auxiliar, nunca autoridad principal. |

Una afirmación como "el score usa percentiles robustos" pertenece a la capa A y
debe verificarse en código. Una afirmación como "la diversificación depende de
correlaciones" pertenece a la capa B/C y debe citar una fuente profesional o
académica.

## Reglas De Navegación

Cada módulo debe decir explícitamente:

- Qué documento se lee primero.
- Qué documento se lee para entender el concepto.
- Qué documento se lee para modificar código.
- Qué documento se lee para usar el módulo.
- Qué documento se lee para consultar parámetros, atributos y métodos.

El `README.md` es la puerta de entrada, pero el inicio real debe ser
`00_index.md`.

## Definición De Hecho

Un hecho documental se considera válido si cumple una de estas condiciones:

- Está respaldado por el código actual.
- Está respaldado por un test.
- Está respaldado por una fuente académica, profesional u oficial.
- Está marcado explícitamente como interpretación, recomendación o supuesto.

Si una afirmación financiera mezcla teoría y código, debe mostrar ambas capas:
fundamento conceptual y traducción concreta a clases, métodos, atributos o
salidas del proyecto.

## Criterios De Calidad

Un módulo está bien documentado cuando una persona nueva puede responder:

- Qué problema resuelve.
- Qué no resuelve.
- Qué datos espera.
- Qué objetos produce.
- Qué métodos implementa.
- Qué teoría o práctica profesional justifica esos métodos.
- Qué supuestos y límites tiene.
- Qué documento debe leer después.

