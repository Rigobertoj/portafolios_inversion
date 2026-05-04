# Plantilla Para Documentar Un Módulo

## Índice Del Documento

1. Propósito
2. Carpeta Esperada
3. Documentos Obligatorios
4. Documentos Opcionales
5. Tabla De Lectura
6. Criterio De Adaptación

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

Esta plantilla define la estructura mínima para documentar un módulo de código.
Debe funcionar para módulos financieros, pipelines de datos, sistemas de
backtesting, APIs internas, modelos estadísticos o cualquier componente con
responsabilidades claras.

## Carpeta Esperada

```text
docs/<module>/
├── README.md
├── 00_index.md
├── 01_conceptual_model.md
├── 02_architecture.md
├── 03_data_contracts.md
├── 04_<primary_article>.md
├── 05_<secondary_article>.md
├── 06_implementation_notes.md
├── 07_workflows.md
├── 08_api_reference.md
├── 09_validation_and_edge_cases.md
├── 10_glossary.md
└── diagrams/
```

## Documentos Obligatorios

| Documento | Pregunta Que Responde |
|---|---|
| `README.md` | Qué es el módulo y dónde empieza la lectura. |
| `00_index.md` | Cuál es el orden recomendado de lectura. |
| `01_conceptual_model.md` | Qué problema conceptual resuelve. |
| `02_architecture.md` | Cómo se organiza internamente. |
| `03_data_contracts.md` | Qué entra, qué sale y con qué forma. |
| `07_workflows.md` | Cómo se usa en escenarios reales. |
| `08_api_reference.md` | Qué clases, funciones, métodos y atributos existen. |
| `09_validation_and_edge_cases.md` | Qué errores, supuestos y límites deben vigilarse. |
| `10_glossary.md` | Cómo se define el vocabulario del módulo. |

## Documentos Opcionales

Los artículos `04_*` y `05_*` son obligatorios cuando el módulo implementa un
método que necesita interpretación. En un módulo simple pueden omitirse, pero en
módulos de finanzas cuantitativas deben existir.

Ejemplos:

- `04_mean_variance_article.md`
- `05_postmodern_downside_article.md`
- `04_fundamental_selection_article.md`
- `05_correlation_selection_article.md`

## Tabla De Lectura

Cada `00_index.md` debe incluir una tabla con esta forma:

| Orden | Documento | Rol | Cuándo Leerlo |
|---:|---|---|---|
| 00 | `00_index.md` | Ruta principal | Siempre primero. |
| 01 | `01_conceptual_model.md` | Concepto | Antes de modificar o usar el módulo. |
| 02 | `02_architecture.md` | Implementación | Antes de tocar código. |
| 07 | `07_workflows.md` | Uso | Cuando se necesita ejecutar un caso real. |
| 08 | `08_api_reference.md` | Consulta | Cuando se busca un método o atributo. |

## Criterio De Adaptación

La plantilla no obliga a que todos los módulos tengan la misma longitud. Obliga
a que tengan la misma lógica. Un módulo pequeño puede tener documentos breves,
pero no debe ocultar la ruta de lectura ni mezclar explicación conceptual con
referencia API sin separación.

