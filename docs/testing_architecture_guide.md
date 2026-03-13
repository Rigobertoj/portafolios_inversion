# Guía pedagógica de testing aplicada a este proyecto

Esta guía toma la estructura actual del repositorio para enseñar **fundamentos, arquitectura y diseño de pruebas** en un proyecto cuantitativo con foco en utilidades de portafolios (`src/portfolio_utils`).

## 1) Diagnóstico del proyecto (visión de testing)

Patrones observados:

- El código de negocio reusable vive en `src/portfolio_utils`.
- Hay notebooks en `src/01` y `src/02` (útiles para exploración, no ideales como base principal de tests).
- La carpeta `test/` existe pero estaba prácticamente vacía para pruebas ejecutables.

**Conclusión pedagógica:**
Para escalar, conviene separar claramente:

1. **Lógica de dominio testeable** (`src/portfolio_utils`).
2. **Exploración/análisis narrativo** (notebooks).
3. **Pruebas automatizadas estables** (`test/portfolio_utils`).

---

## 2) Pirámide de testing recomendada para tu escala

Para este repositorio (escala académica/prototipo avanzado), una pirámide realista es:

- **70% unitarias**: validan funciones y métodos aislados (`AssetsResearch`, métricas, transformaciones).
- **20% integración liviana**: validan flujo entre módulos y contratos de datos (DataFrame de entrada/salida).
- **10% end-to-end**: opcional, para pipelines completos o scripts reproducibles.

> Regla práctica: mientras más dependencias externas (APIs, red, archivos grandes), menos deben dominar tus tests del día a día.

---

## 3) Diseño de pruebas por capas

## Capa A — Unit tests (rápidos y deterministas)

Objetivo: probar reglas matemáticas y validaciones sin red.

Ejemplos en este proyecto:

- validación de constructor (`tickers`, `start`).
- cálculo de `compute_returns` con datos en cache.
- estructura de columnas de `metrics`.

Técnicas clave:

- usar datos sintéticos pequeños (4–10 filas).
- comparar con cálculo manual (`mean`, `std`, anualización).
- evitar IO real.

## Capa B — Integración (contratos entre módulos)

Objetivo: confirmar que módulos colaboran con forma de datos consistente.

Ideas para siguientes iteraciones:

- integrar `AssetsResearch` + `PortfolioElementaryMetrics` con un DataFrame conocido.
- validar que la matriz de covarianza/correlación mantiene índices/columnas esperadas.

## Capa C — E2E académico (opcional)

Objetivo: verificar un flujo “desde datos hasta métrica final”.

Ejemplo:

1. descargar/preparar precios (con fixture local, no red real).
2. construir retornos.
3. ejecutar optimización o reporte.
4. validar métricas clave y no-regresiones.

---

## 4) Estrategia contra fragilidad en proyectos financieros

En finanzas cuantitativas, pruebas frágiles suelen venir de:

- dependencia de internet (`yfinance`).
- cambios de frecuencia/fechas.
- diferencias numéricas por redondeo.

Buenas prácticas:

- **mock de proveedores externos** en unit tests.
- tolerancia numérica (`np.testing.assert_allclose`).
- fixtures de datos congelados para reproducibilidad.

---

## 5) Convenciones sugeridas de arquitectura de tests

Estructura recomendada:

```text
test/
  conftest.py
  portfolio_utils/
    test_assets_research.py
    test_portfolio_elementary_metrics.py
    test_portfolio_optimization.py
  integration/
    test_pipeline_metrics.py
```

Convenciones:

- nombre: `test_<modulo>.py`.
- patrón AAA (Arrange, Act, Assert).
- un comportamiento por test.
- mensajes de error explícitos.

---

## 6) Qué se mejoró ahora y por qué

Se implementaron pruebas unitarias en `test/portfolio_utils/test_assets_research.py` para cubrir:

- validación de parámetros de entrada.
- cálculo determinista de retornos.
- mock de descarga `yfinance` con `monkeypatch`.
- validación de errores al solicitar tickers inexistentes.
- validación de métricas contra cálculo manual.

También se agregó `test/conftest.py` para asegurar importación limpia desde `src/` en ejecución de `pytest`.

---

## 7) Hoja de ruta pedagógica (4 semanas)

Semana 1:

- dominar unit tests de `AssetsResearch`.
- cubrir happy path + edge cases (NaN, series cortas, retornos cero).

Semana 2:

- crear tests para `PortfolioElementaryMetrics`.
- introducir parametrización con `pytest.mark.parametrize`.

Semana 3:

- tests de integración entre módulos.
- fixtures versionadas (CSV pequeños en `test/data`).

Semana 4:

- medir cobertura (`pytest --cov`).
- definir smoke tests para pipeline general.

---

## 8) Checklist de calidad para cada nuevo módulo

Antes de cerrar un módulo, verifica:

- [ ] valida entradas inválidas con excepciones claras.
- [ ] output con forma estable (índices/columnas documentadas).
- [ ] cálculos numéricos con tolerancias adecuadas.
- [ ] tests sin red ni dependencias no deterministas.
- [ ] al menos 1 test de comportamiento límite.

---

## 9) Criterio de “buen test” en tu contexto

Un buen test en este proyecto debe ser:

- **rápido** (< 200 ms en unitario típico).
- **determinista** (mismo resultado siempre).
- **legible** (explica el negocio financiero, no solo la sintaxis).
- **útil para refactorizar** (si rompes un contrato real, te avisa).

Si quieres, en una siguiente iteración puedo ayudarte a crear:

1. matriz de cobertura objetivo por módulo,
2. plantilla base de tests para cada clase,
3. pipeline de CI para ejecutar pruebas automáticamente en cada commit.
