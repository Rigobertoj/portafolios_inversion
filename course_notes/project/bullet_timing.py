import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# DATOS — TIIE y tasas trimestrales 2016-Q2 → 2026-Q2
# ============================================================
TRIMESTRES = [
    '2016-Q2','2016-Q3','2016-Q4',
    '2017-Q1','2017-Q2','2017-Q3','2017-Q4',
    '2018-Q1','2018-Q2','2018-Q3','2018-Q4',
    '2019-Q1','2019-Q2','2019-Q3','2019-Q4',
    '2020-Q1','2020-Q2','2020-Q3','2020-Q4',
    '2021-Q1','2021-Q2','2021-Q3','2021-Q4',
    '2022-Q1','2022-Q2','2022-Q3','2022-Q4',
    '2023-Q1','2023-Q2','2023-Q3','2023-Q4',
    '2024-Q1','2024-Q2','2024-Q3','2024-Q4',
    '2025-Q1','2025-Q2','2025-Q3','2025-Q4',
    '2026-Q1','2026-Q2',
]

TIIE = [
     3.75, 4.25, 5.25,
     6.25, 6.75, 7.00, 7.25,
     7.50, 7.75, 7.75, 8.25,
     8.25, 8.25, 7.75, 7.25,
     6.50, 5.00, 4.50, 4.25,
     4.00, 4.25, 4.75, 5.50,
     6.50, 7.75, 9.25,10.50,
    11.25,11.25,11.25,11.25,
    11.00,10.75,10.25,10.00,
     9.00, 8.50, 7.75, 7.25,
     7.00, 6.50,
]

CETES28 = [
     3.77, 4.31, 5.33,
     6.32, 6.82, 7.05, 7.30,
     7.55, 7.80, 7.81, 8.33,
     8.28, 8.26, 7.79, 7.27,
     6.53, 5.03, 4.52, 4.27,
     4.01, 4.26, 4.80, 5.56,
     6.56, 7.80, 9.34,10.55,
    11.29,11.31,11.33,11.29,
    11.04,10.79,10.29,10.03,
     9.03, 8.53, 7.79, 7.27,
     7.01, 6.49,
]

CETES364 = [
     4.20, 4.85, 5.95,
     6.95, 7.35, 7.55, 7.80,
     8.05, 8.30, 8.30, 8.90,
     8.80, 8.75, 8.25, 7.75,
     7.00, 5.40, 4.85, 4.55,
     4.30, 4.60, 5.20, 6.10,
     7.20, 8.50,10.05,11.00,
    11.65,11.68,11.70,11.60,
    11.30,11.05,10.55,10.28,
     9.25, 8.80, 8.05, 7.55,
     7.25, 7.00,
]

BONO10 = [
     6.02, 6.35, 7.37,
     7.79, 7.12, 7.11, 7.74,
     7.84, 7.98, 8.10, 9.01,
     8.26, 8.20, 7.54, 7.00,
     8.05, 6.48, 6.13, 5.97,
     6.98, 7.80, 7.45, 7.72,
     8.20, 9.36,10.03,10.18,
     9.11, 9.10,10.24, 9.77,
     9.37,10.36, 9.87, 9.90,
     9.46, 9.65, 9.37, 9.37,
     9.19, 9.07,
]

# ============================================================
# PARÁMETROS  <-- modifica aquí
# ============================================================
CAPITAL          = 100_000
ISR_Q            = 0.0090 / 4
DUR_10A          = 7.3

UMBRAL_ALTO      = 8.50   # TIIE >= aquí → máxima exposición al Bono10a
UMBRAL_MEDIO     = 7.00   # TIIE >= aquí → exposición media al Bono10a
MAX_PESO_BONO    = 0.85   # nunca más del 85% en Bono10a               [NUEVA]
TAKE_PROFIT      = 0.10   # vender si ganancia de precio >= 10%
CETES28_MINIMO   = 5.00   # si Cetes28 < 5% → salir de bonos largos    [NUEVA]

# ============================================================
# BACKTEST
# ============================================================
cap_bullet = CAPITAL
cap_cetes  = CAPITAL

tc10       = 0.0    # tasa de compra del Bono10a (0 = sin posición)
w10        = 0.0    # peso en Bono10a
wc364      = 1.0    # peso en Cetes364

hist_bullet = [cap_bullet]
hist_cetes  = [cap_cetes]
eventos     = []    # (índice, tipo) para la gráfica

for i in range(1, len(TRIMESTRES)):
    tiie  = TIIE[i]
    c28   = CETES28[i]
    c364  = CETES364[i]
    b10   = BONO10[i]

    # ----------------------------------------------------------
    # REGLA NUEVA 3: Cetes28 < 5% → entorno de tasa baja,
    # el bono largo ya no compensa el riesgo → salir
    # ----------------------------------------------------------
    if c28 < CETES28_MINIMO and w10 > 0:
        w10 = 0.0;  wc364 = 1.0
        eventos.append((i, 'salida_tasa_baja'))

    # ----------------------------------------------------------
    # REGLAS DE ENTRADA (solo si no estamos en tasa baja)
    # ----------------------------------------------------------
    elif c28 >= CETES28_MINIMO:
        if tiie >= UMBRAL_ALTO and w10 < MAX_PESO_BONO:
            # Máxima exposición respetando el techo del 85%    [NUEVA: cap 85%]
            w10   = MAX_PESO_BONO
            wc364 = 1.0 - MAX_PESO_BONO
            tc10  = b10
            eventos.append((i, 'compra'))

        elif tiie >= UMBRAL_MEDIO and w10 < 0.45:
            # Exposición media: 50% pero sin superar 85%
            w10   = min(0.50, MAX_PESO_BONO)
            wc364 = 1.0 - w10
            tc10  = b10
            eventos.append((i, 'compra'))

        elif tiie < UMBRAL_MEDIO and w10 > 0:
            # TIIE bajó del umbral mínimo → salir de bonos
            w10 = 0.0;  wc364 = 1.0

    # ----------------------------------------------------------
    # TAKE-PROFIT: ganancia de precio >= 10%                   [NUEVA: siempre]
    # ----------------------------------------------------------
    ganancia_extra = 0.0
    if w10 > 0 and tc10 > 0:
        gp = -DUR_10A * (b10 - tc10) / 100
        if gp >= TAKE_PROFIT:
            ganancia_extra = w10 * gp   # materializar ganancia
            tc10  = b10                 # reinvertir al rendimiento vigente
            eventos.append((i, 'take_profit'))

    # ----------------------------------------------------------
    # RETORNO DEL TRIMESTRE
    # ----------------------------------------------------------
    r_bullet = (wc364 * (c364 / 100 / 4) +
                w10   * (tc10  / 100 / 4 if tc10 else 0) +
                ganancia_extra - ISR_Q)

    r_cetes = c28 / 100 / 4 - ISR_Q

    cap_bullet *= (1 + r_bullet)
    cap_cetes  *= (1 + r_cetes)

    hist_bullet.append(round(cap_bullet, 2))
    hist_cetes.append(round(cap_cetes, 2))

# ============================================================
# RESULTADOS
# ============================================================
n_años = (len(TRIMESTRES) - 1) / 4
cagr_b = ((hist_bullet[-1] / CAPITAL) ** (1/n_años) - 1) * 100
cagr_c = ((hist_cetes[-1]  / CAPITAL) ** (1/n_años) - 1) * 100

print(f"{'':30} {'Bullet v2':>12} {'Cetes 28d':>12}")
print(f"{'Capital final':30} ${hist_bullet[-1]:>11,.0f} ${hist_cetes[-1]:>11,.0f}")
print(f"{'CAGR anual neto':30} {cagr_b:>11.2f}% {cagr_c:>11.2f}%")
print(f"{'Ganancia total':30} ${hist_bullet[-1]-CAPITAL:>11,.0f} ${hist_cetes[-1]-CAPITAL:>11,.0f}")
print(f"{'Alpha':30} {cagr_b - cagr_c:>+11.2f}%")
print()
n_compras    = sum(1 for _, t in eventos if t == 'compra')
n_tp         = sum(1 for _, t in eventos if t == 'take_profit')
n_baja       = sum(1 for _, t in eventos if t == 'salida_tasa_baja')
print(f"Compras ejecutadas:          {n_compras}")
print(f"Take-profits ejecutados:     {n_tp}")
print(f"Salidas por tasa baja (<5%): {n_baja}")

# ============================================================
# GRÁFICA
# ============================================================
x      = range(len(TRIMESTRES))
ticks  = list(range(0, len(TRIMESTRES), 4))
labels = [TRIMESTRES[i][:4] for i in ticks]

fig, ax = plt.subplots(figsize=(13, 6), facecolor='white')

ax.plot(x, [v/1000 for v in hist_cetes],  color='#888780',
        lw=2, ls='--', label='Cetes 28d benchmark')
ax.plot(x, [v/1000 for v in hist_bullet], color='#1D9E75',
        lw=2.5, label='Bullet con timing v2')

ax.fill_between(x,
    [e/1000 for e in hist_bullet],
    [c/1000 for c in hist_cetes],
    where=[e >= c for e, c in zip(hist_bullet, hist_cetes)],
    alpha=0.12, color='#1D9E75')
ax.fill_between(x,
    [e/1000 for e in hist_bullet],
    [c/1000 for c in hist_cetes],
    where=[e < c for e, c in zip(hist_bullet, hist_cetes)],
    alpha=0.12, color='#D85A30')

# Marcadores de eventos
colores = {'compra': '#185FA5', 'take_profit': '#D85A30', 'salida_tasa_baja': '#BA7517'}
marcadores = {'compra': '^', 'take_profit': 'v', 'salida_tasa_baja': 'X'}
etiquetas  = {'compra': 'Compra Bono10a', 'take_profit': 'Take-profit 10%',
              'salida_tasa_baja': 'Salida (C28 < 5%)'}

vistos = set()
for idx, tipo in eventos:
    label = etiquetas[tipo] if tipo not in vistos else '_nolegend_'
    vistos.add(tipo)
    ax.scatter(idx, hist_bullet[idx]/1000,
               marker=marcadores[tipo], s=100,
               color=colores[tipo], zorder=5, label=label)

ax.annotate(f'${hist_bullet[-1]:,.0f}  ({cagr_b:.2f}% CAGR)',
            xy=(len(TRIMESTRES)-1, hist_bullet[-1]/1000),
            xytext=(-6, 6), textcoords='offset points',
            ha='right', color='#1D9E75', fontsize=9, fontweight='bold')
ax.annotate(f'${hist_cetes[-1]:,.0f}  ({cagr_c:.2f}% CAGR)',
            xy=(len(TRIMESTRES)-1, hist_cetes[-1]/1000),
            xytext=(-6, -14), textcoords='offset points',
            ha='right', color='#888780', fontsize=9, fontweight='bold')

ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Capital ($k MXN)', fontsize=10)
ax.set_title(
    'Backtest: Bullet con timing v2  vs  Cetes 28d  |  may-2016 → may-2026\n'
    'Reglas: TIIE ≥ 8.5% → 85% Bono10a  |  TIIE ≥ 7% → 50% Bono10a  |  '
    'Take-profit 10%  |  Salir si Cetes28 < 5%',
    fontsize=9.5)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('bullet_v2_resultado.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Gráfica guardada: bullet_v2_resultado.png")