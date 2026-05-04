# Plantilla Para Artículos Explicativos

## Índice Del Documento

1. Propósito
2. Cuándo Usar Un Artículo
3. Estructura Obligatoria
4. Regla De Traducción Teoría-Código
5. Ejemplo De Bloque Interpretativo

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

Un artículo explicativo convierte un método implementado en una pieza legible
para una persona que necesita entender el concepto, el diseño y las
implicaciones prácticas.

No reemplaza la API reference. La complementa.

## Cuándo Usar Un Artículo

Se debe escribir un artículo cuando el módulo contiene:

- Una teoría financiera o estadística.
- Un modelo de scoring.
- Un optimizador.
- Un método con supuestos relevantes.
- Una relación no trivial entre datos, parámetros, estados y salidas.
- Un concepto que pueda usarse mal si solo se lee la firma de métodos.

## Estructura Obligatoria

```text
# Título Del Artículo

## Índice Del Documento
## Mapa De La Serie
## Problema Que Resuelve
## Fundamento Conceptual
## Fuentes Académicas O Profesionales
## Traducción Al Código
## Clases, Métodos Y Atributos Implicados
## Ejemplo Interpretado
## Supuestos Y Límites
## Errores Comunes
## Relación Con Workflows
## Referencias
```

## Regla De Traducción Teoría-Código

Cada concepto relevante debe aterrizar en el código:

| Concepto | Traducción Esperada |
|---|---|
| Tesis de inversión | Configuración, pesos, métricas y dirección de score. |
| Riesgo | Métrica, matriz, vector, threshold o resultado concreto. |
| Universo | Entrada, normalización, filtro o contrato de datos. |
| Selección | Método que ordena, filtra o produce candidatos. |
| Interpretabilidad | Reporte, columnas de score, cobertura o diagnóstico. |

## Ejemplo De Bloque Interpretativo

Una descripción pobre dice:

```text
score_coverage : float
Cobertura del score.
```

Una descripción aceptable dice:

```text
score_coverage mide qué proporción del peso total de la configuración tuvo datos
disponibles para una compañía. No es una medida de calidad financiera; es una
medida de completitud del score. Debe revisarse junto con `fundamental_score`
porque una empresa puede obtener buen score con baja cobertura si muchos
componentes no estaban disponibles.
```

