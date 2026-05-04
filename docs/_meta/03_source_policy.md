# Política De Fuentes

## Índice Del Documento

1. Propósito
2. Jerarquía De Fuentes
3. Fuentes Recomendadas En Finanzas
4. Reglas De Citación
5. Regla Para Código Y Teoría

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

Esta política evita que los artículos conceptuales dependan de explicaciones
sueltas o de intuiciones sin respaldo. La documentación puede tener tono
explicativo, pero sus afirmaciones centrales deben estar ancladas.

## Jerarquía De Fuentes

| Prioridad | Fuente | Ejemplo | Uso |
|---:|---|---|---|
| 1 | Código y tests del repo | `src/selection/fundamental_scorers.py` | Verdad sobre lo implementado. |
| 2 | Papers o libros académicos | Markowitz, Sharpe, Fama-French | Fundamento teórico. |
| 3 | Organismos profesionales | CFA Institute, GIPS | Interpretación profesional. |
| 4 | Documentación oficial | pandas, NumPy, SciPy, yfinance | Comportamiento técnico. |
| 5 | Material secundario | Blogs, cursos, notas | Apoyo contextual. |

## Fuentes Recomendadas En Finanzas

Para selección fundamental:

- CFA Institute, *Equity Valuation: Concepts and Basic Tools*.
- CFA Institute, *Active Equity Investing: Strategies*.
- CFA Institute, *Market-Based Valuation: Price and Enterprise Value Multiples*.

Para selección por correlación y diversificación:

- CFA Institute, *Portfolio Risk and Return: Part I*.
- Harry Markowitz, *Portfolio Selection*, Journal of Finance, 1952.

Para optimización y riesgo:

- CFA Institute, *Principles of Asset Allocation*.
- CFA Institute, *The Sortino Ratio: Is Downside Risk the Only Risk that Matters?*
- Keating y Shadwick, *A Universal Performance Measure*, 2002, para Omega.

## Reglas De Citación

Cada artículo debe incluir una sección `Referencias` con:

- Nombre de la fuente.
- Institución o autor.
- URL cuando exista.
- Fecha de consulta si la fuente web puede cambiar.
- Nota breve de por qué se usa.

No se deben copiar fragmentos largos. La documentación debe parafrasear,
interpretar y conectar la fuente con el código.

## Regla Para Código Y Teoría

Cuando una afirmación dependa de código y teoría, se deben citar ambas capas.

Ejemplo:

```text
La selección por correlación se apoya conceptualmente en diversificación y
correlación entre activos; en el código, esa idea se traduce en `_compute_corr_scores`,
que rankea tickers por correlación promedio contra pares.
```

