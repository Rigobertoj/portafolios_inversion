from pathlib import Path


OUT = Path(__file__).resolve().parent


PREAMBLE = r"""\documentclass[12pt,letterpaper]{article}

\usepackage{fontspec}
\setmainfont[
  Path=/mnt/c/Windows/Fonts/,
  UprightFont=GARA.TTF,
  BoldFont=GARABD.TTF,
  ItalicFont=GARAIT.TTF
]{Garamond}
\usepackage[margin=0.82in]{geometry}
\usepackage{setspace}
\setstretch{1.25}
\usepackage{ragged2e}
\justifying
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{float}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{tocloft}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{lastpage}

\definecolor{gsnavy}{HTML}{1E3D70}
\definecolor{gsblue}{HTML}{7E93B0}
\definecolor{gspale}{HTML}{D8DEE7}
\definecolor{gsdark}{HTML}{0A3850}
\definecolor{softgray}{HTML}{F2F3F5}
\definecolor{textgray}{HTML}{343A40}

\hypersetup{colorlinks=true, linkcolor=gsnavy, urlcolor=gsnavy}
\pagestyle{fancy}
\fancyhf{}
\lhead{\textcolor{gsblue}{<<TICKER>> | <<PHILOSOPHY>> Investment Memo}}
\rhead{\textcolor{gsblue}{Mayo 2026}}
\cfoot{\textcolor{gsblue}{\thepage\ de \pageref{LastPage}}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\contentsname}{Contents}
\sloppy
\emergencystretch=3em
\setlength{\headheight}{15pt}

\titleformat{\section}{\Large\bfseries\color{gsnavy}}{\thesection.}{0.6em}{}
\titleformat{\subsection}{\large\bfseries\color{gsnavy}}{\thesubsection}{0.6em}{}
\setlist[itemize]{leftmargin=1.1em, itemsep=0.2em, topsep=0.2em}
\renewcommand{\cfttoctitlefont}{\Large\bfseries\color{gsnavy}}
\renewcommand{\cftsecfont}{\bfseries\color{gsnavy}}
\renewcommand{\cftsecpagefont}{\bfseries}
\renewcommand{\cftsecleader}{\hfill}
\renewcommand{\cftsecaftersnum}{\hspace{0.9em}}
\setlength{\cftsecindent}{0pt}
\setlength{\cftsecnumwidth}{2.0em}
\setlength{\cftbeforesecskip}{1.15em}
\setlength{\cftbeforetoctitleskip}{0.25em}
\setlength{\cftaftertoctitleskip}{1.35em}
\renewcommand{\arraystretch}{1.14}
\setlength{\tabcolsep}{8pt}

\newcommand{\tabletop}{\arrayrulecolor{textgray}\toprule}
\newcommand{\tablemid}{\arrayrulecolor{textgray}\midrule}
\newcommand{\tablebottom}{\arrayrulecolor{textgray}\bottomrule}

\newcommand{\metricbox}[3]{
\begin{minipage}[t]{0.31\textwidth}
\vspace{0pt}
\colorbox{gsnavy}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{\centering\color{white}\bfseries #1}}\\[-0.1em]
\fcolorbox{gsnavy}{white}{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\vspace{0.45em}\centering{\Large\bfseries\color{gsnavy} #2}\\{\small #3}\vspace{0.45em}}}
\end{minipage}
}

\begin{document}
"""


TITLE = r"""
\begin{titlepage}
\thispagestyle{empty}
\begin{tikzpicture}[remember picture,overlay]
  \fill[white] (current page.south west) rectangle (current page.north east);
  \fill[gsblue] (current page.south west) rectangle ([xshift=0.62in]current page.north west);
  \fill[gspale] ([yshift=-1.55in]current page.north west) rectangle ([yshift=-2.25in]current page.north east);
  \fill[gsnavy] ([xshift=-1.25in,yshift=-1.55in]current page.north east) rectangle ([xshift=-0.70in,yshift=-2.25in]current page.north east);
  \fill[gsdark] ([xshift=0.62in,yshift=-2.25in]current page.north west) rectangle ([xshift=-0.55in,yshift=-6.45in]current page.north east);
  \node[anchor=west,text=white,font=\Huge\bfseries] at ([xshift=0.95in,yshift=-3.15in]current page.north west) {<<COMPANY>>};
  \node[anchor=east,text=white,font=\large] at ([xshift=-0.95in,yshift=-3.15in]current page.north east) {10 de mayo de 2026};
  \node[anchor=west,text=white,font=\LARGE] at ([xshift=0.95in,yshift=-3.65in]current page.north west) {<<PHILOSOPHY>> Investment Memo};
  \node[anchor=east,text=white,font=\large] at ([xshift=-0.95in,yshift=-3.65in]current page.north east) {<<EXCHANGE>>: <<TICKER>>};
  \node[anchor=west,text=white,font=\normalsize,text width=5.75in,align=left] at ([xshift=0.95in,yshift=-4.10in]current page.north west) {Security selection | <<SECTOR>>};
  \node[anchor=east,text=white,font=\normalsize] at ([xshift=-0.95in,yshift=-4.55in]current page.north east) {Filosofia: <<PHILOSOPHY>>};
  \node[anchor=west,text=gsnavy,font=\bfseries] at ([xshift=0.95in,yshift=-9.85in]current page.north west) {Decision preview};
  \node[anchor=west,text=gsnavy,font=\normalsize] at ([xshift=0.95in,yshift=-10.12in]current page.north west) {<<PREVIEW>>};
\end{tikzpicture}
\end{titlepage}

\tableofcontents
\thispagestyle{fancy}
\newpage
"""


def table(headers, rows, widths="lXXX", size="small"):
    head = " & ".join([rf"\textbf{{{h}}}" for h in headers]) + r"\\"
    body = "\n".join([" & ".join(row) + r"\\" for row in rows])
    return rf"""
\begin{{table}}[H]
\centering
\{size}
\begin{{tabularx}}{{\textwidth}}{{{widths}}}
\tabletop
{head}
\tablemid
{body}
\tablebottom
\end{{tabularx}}
\end{{table}}
"""


def metric_boxes(metrics):
    lines = []
    for i in range(0, len(metrics), 3):
        chunk = metrics[i:i + 3]
        boxes = "\n\\hfill\n".join([rf"\metricbox{{{a}}}{{{b}}}{{{c}}}" for a, b, c in chunk])
        lines.append("\\noindent\n" + boxes + "\n\n\\vspace{1.0em}")
    return "\n\n".join(lines)


def score_table(rows):
    return table(
        ["Dimension metodologica", "Peso", "Score", "Lectura de comite"],
        rows,
        widths="lrrX",
        size="scriptsize",
    )


def render(memo):
    tex = PREAMBLE.replace("<<TICKER>>", memo["ticker"]).replace("<<PHILOSOPHY>>", memo["philosophy"])
    tex += TITLE
    for k in ["company", "philosophy", "exchange", "ticker", "sector", "preview"]:
        tex = tex.replace(f"<<{k.upper()}>>", memo[k])
    tex += memo["body"]
    tex += "\n\\end{document}\n"
    (OUT / memo["file"]).write_text(tex, encoding="utf-8")


def interpolate(template):
    """Evaluate only the helper placeholders embedded in the TeX template."""
    allowed = ("metric_boxes(", "table(", "score_table(")
    out = []
    i = 0
    while i < len(template):
        if template[i] == "{" and any(template.startswith(name, i + 1) for name in allowed):
            j = i + 1
            depth = 0
            quote = None
            escaped = False
            while j < len(template):
                ch = template[j]
                if quote:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == quote:
                        quote = None
                else:
                    if ch in ("'", '"'):
                        quote = ch
                    elif ch in "([{":
                        depth += 1
                    elif ch in ")]}":
                        depth -= 1
                        if depth < 0:
                            break
                j += 1
            expr = template[i + 1:j]
            out.append(str(eval(expr, globals(), locals())))
            i = j + 1
        else:
            out.append(template[i])
            i += 1
    return "".join(out).replace("\\\\&", "\\&").replace("\\\\%", "\\%")


well_common = {
    "company": "Welltower Inc. (WELL)",
    "ticker": "WELL",
    "exchange": "NYSE",
    "sector": "Healthcare real estate, seniors housing \\& outpatient medical",
}

axp_common = {
    "company": "American Express Company (AXP)",
    "ticker": "AXP",
    "exchange": "NYSE",
    "sector": "Payments, premium consumer finance \\& network services",
}

wdc_common = {
    "company": "Western Digital Corporation (WDC)",
    "ticker": "WDC",
    "exchange": "NASDAQ",
    "sector": "Technology hardware, cloud storage \\& HDD infrastructure",
}


def well_value_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Market Cap","USD 151.5bn","FactSet Snapshot, cierre 8-may-2026"),
("EV/EBITDA LTM","60.9x","Multiplo incompatible con Value estricto"),
("FCF Yield","1.5\\%","FCF Q1 annualizado sobre capitalizacion"),
("S\\&P Rating","A-","FactSet DCS, outlook estable"),
("Net Debt/EBITDA","5.3x","Mejora desde 6.3x en dic-2025"),
("Score Value","48/100","No inclusion bajo Value")
])}

Welltower no clasifica como activo Value invertible en el punto actual de entrada. La compania es de alta calidad operativa dentro de healthcare real estate, con exposicion estructural a seniors housing, recuperacion de ocupacion, pricing power en RevPOR y una plataforma de capital allocation que ha producido crecimiento superior. Sin embargo, el lente Value exige margen de seguridad, no solo calidad del activo. A EV/EBITDA LTM de 60.9x, P/S LTM de 13.3x y FCF yield cercana a 1.5\%, el mercado ya descuenta una porcion sustancial del crecimiento esperado.

El dictamen para comite es claro: \textbf{rechazar inclusion bajo filosofia Value y mantener en watchlist por precio}. La tesis fundamental puede ser atractiva para Quality o Growth-at-a-reasonable-price, pero bajo Value el activo presenta el riesgo clasico de pagar demasiado por un negocio correcto. La recuperacion de seniors housing y la escasez de oferta son reales; el problema es que el precio incorpora una narrativa muy favorable y reduce el margen de error ante tasas, cap rates, ejecucion de adquisiciones o moderacion de NOI.

\section{Investment View}

Welltower fue fundada en 1970 y opera como un REIT especializado en infraestructura inmobiliaria de salud y vivienda para adultos mayores. Cotiza en NYSE bajo el ticker WELL y participa en seniors housing operating, triple-net senior housing y outpatient medical. Su mercado geografico central es Estados Unidos, con presencia adicional en Canada y Reino Unido. La marca no se posiciona como operador tradicional de inmuebles, sino como una plataforma de real estate, datos, operadores asociados y capital allocation.

La lectura cualitativa es favorable: Welltower se ubica en el centro de la llamada silver economy, con una cartera de comunidades de seniors housing y wellness housing en micromercados atractivos. La tesis operativa se apoya en envejecimiento poblacional, baja construccion nueva, recuperacion de ocupacion y capacidad para adquirir activos a bases atractivas cuando el mercado de capitales inmobiliario esta dislocado. Estos elementos crean un negocio estructuralmente superior al REIT promedio.

La lectura Value, sin embargo, cambia la conclusion. Un activo puede ser excelente y aun asi no ser comprable bajo disciplina de margen de seguridad. La compania reporta mejora de balance, mayor guia de FFO y crecimiento organicamente fuerte, pero la valuacion esta mas cerca de un activo de crecimiento secular que de un valor infravalorado. El comite debe separar "empresa de calidad" de "precio Value"; en WELL, la primera condicion se cumple, la segunda no.

\section{Business Model \& Cash Engine}

El modelo de generacion de valor combina renta inmobiliaria, NOI operativo, adquisiciones, redevelopment, joint ventures y acceso a capital. El segmento mas importante para la narrativa actual es Seniors Housing Operating, donde Welltower se beneficia directamente de ocupacion, RevPOR, eficiencia operativa y recuperacion de margen. Outpatient Medical aporta estabilidad, leases medicos y menor volatilidad relativa. Triple-net funciona como capa contractual, aunque con menor optionality operativa.

La diferencia frente a un REIT pasivo es que Welltower busca operar como una empresa de operaciones dentro de una envoltura inmobiliaria. Su Welltower Business System, relaciones con operadores y plataforma de datos pretenden convertir seleccion de micromercados, densidad regional y ejecucion operativa en alpha. Esa arquitectura es importante, pero bajo Value hay que preguntar cuanto de ese alpha ya esta capitalizado.

{table(["Motor", "Dato", "Lectura Value", "Implicacion"], [
["SHO same-store NOI", "22.1\\% Q1 26", "Crecimiento excepcional", "No basta si el precio ya descuenta varios anos de recuperacion."],
["Revenue same-store", "9.5\\%", "Demanda y precio", "Valida el activo, no el margen de seguridad."],
["Ocupacion SHO", "+370 bps", "Recuperacion ciclica", "Riesgo si la normalizacion pierde velocidad."],
["Guidance FFO", "USD 6.28", "Mejora de 11c", "Catalizador positivo, pero insuficiente para justificar multiplo actual."],
], widths="lXXX")}

\section{Competitive Position \& Addressable Market}

Welltower compite por activos, operadores, capital y ubicaciones con otros REITs de salud, private equity inmobiliario, fondos de infraestructura, operadores de senior living y plataformas medicas. Su ventaja esta en densidad, acceso a capital, analitica de micromercados, relaciones con operadores y track record de adquisiciones. En seniors housing, la oferta nueva limitada y el crecimiento de poblacion 80+ dan soporte a varios anos de demanda potencial.

El TAM es amplio, pero no ilimitado. El mercado final depende de capacidad de pago de residentes, disponibilidad laboral, costos operativos, cap rates y salud financiera de operadores. Una tesis Value exige comprar esos flujos con descuento frente a valor intrinseco. En WELL, el mercado ya asigna un premium por escasez de oferta, calidad de cartera y plataforma operativa. Ese premium puede ser racional, pero no constituye margen de seguridad.

\section{Macro \& Liquidity Backdrop}

En una desaceleracion ordenada, Welltower tiene dos fuerzas opuestas. Por un lado, seniors housing tiene drivers demograficos menos dependientes del ciclo corto que oficinas, retail o lodging. Por otro lado, un REIT de duration larga es sensible a tasas, cap rates, costo de deuda y disponibilidad de equity. La plomeria financiera importa porque el valor de un REIT se forma tanto por NOI como por el spread entre rendimiento de activos y costo de capital.

El entorno actual ayuda a operadores con acceso a capital, porque la dislocacion de creditos inmobiliarios puede crear oportunidades de adquisicion. Welltower ha anunciado USD 10.5bn de inversiones pro rata year-to-date, lo que confirma capacidad de actuar cuando otros compradores estan restringidos. Para Value, la pregunta es si esas adquisiciones se estan comprando a retornos suficientemente superiores al costo de capital para compensar el precio de la accion. El mercado parece asumir que si; el memo exige evidencia continua antes de pagar.

\section{Financial Profile \& Estimate Quality}

El crecimiento historico y proyectado es robusto. FactSet Snapshot muestra ingresos desde USD 6.6bn en 2023 hacia USD 37.5bn estimados en 2028, EBITDA de USD 2.5bn a USD 6.9bn y FCF de USD 1.2bn a USD 5.2bn. La cobertura de estimaciones es amplia y el consenso esta respaldado por guidance operativo y adquisiciones anunciadas. Aun asi, la calidad de la estimacion depende de dos supuestos sensibles: continuidad del crecimiento NOI en SHO y capacidad de financiar crecimiento externo sin destruir valor por dilucion o deuda cara.

{table(["Metrica", "2023", "2024", "2025", "2026E", "2027E", "2028E"], [
["Ingresos (USD m)", "6,642", "8,119", "13,021", "26,475", "30,958", "37,508"],
["EBITDA (USD m)", "2,518", "2,943", "2,678", "5,306", "6,249", "6,938"],
["EPS diluido", "0.52", "0.82", "-0.77", "2.69", "3.24", "3.72"],
["FCF (USD m)", "1,183", "1,393", "2,108", "3,835", "4,391", "5,156"],
["Debt/EBITDA", "6.4x", "5.7x", "8.0x", "3.4x", "2.9x", "--"],
], widths="lrrrrrr", size="scriptsize")}

El perfil financiero no es debil; el problema es la asimetria. Si el consenso acierta, la accion podria sostenerse. Si el crecimiento se modera, el multiplo actual amplifica downside. Value requiere proteccion cuando la narrativa se enfria. WELL no ofrece esa proteccion hoy.

\section{Margin of Safety \& Value Discipline}

La metodologia Value penaliza tres elementos: ausencia de descuento claro, yield de FCF bajo y dependencia de expansion/compresion de cap rates. WELL presenta buen balance relativo, rating A- y mejoras de leverage, pero esos puntos no compensan una valuacion que se aproxima mas a excelencia capitalizada que a ineficiencia de mercado.

{score_table([
["Margen de seguridad", "20", "3", "No hay descuento evidente; target de consenso y precio ya reflejan recuperacion de SHO."],
["Calidad de flujo y FFO", "15", "11", "FFO y NOI mejoran, aunque FCF yield es bajo frente a riesgo de duration."],
["Rentabilidad y spread", "10", "5", "ROIC contable bajo por naturaleza REIT; creacion depende de cap rates y costo de capital."],
["Balance y solvencia", "15", "10", "A-, liquidez amplia y leverage bajando, pero deuda sigue relevante."],
["Catalizadores", "10", "7", "Guidance de FFO, adquisiciones y SHO NOI sostienen momentum."],
["Riesgo de value trap", "15", "5", "No es value trap operativo; si puede ser valuation trap."],
["Macro y liquidez", "15", "7", "Demografia favorable; tasas y cap rates limitan margen de seguridad."],
])}

\section{Valuation, Reverse DCF \& Credit}

El reverse DCF cualitativo exige que el comprador acepte varios anos de crecimiento de FFO/FCF, ejecucion sostenida en adquisiciones y ausencia de compresion material en multiples. Con EV de USD 172.9bn y EV/EBITDA LTM de 60.9x, el activo requiere una narrativa de compounding prolongado. Esa narrativa puede ocurrir, pero no es un precio Value.

La parte crediticia es mas favorable: S\\&P A-, cash e inversiones de corto plazo de USD 4.8bn, lineas disponibles y Net Debt/EBITDA de 5.3x frente a 6.3x al cierre de 2025. La mejora de balance reduce riesgo de insolvencia, pero no crea upside suficiente en equity. Para Value, solvencia es condicion necesaria; precio con descuento sigue siendo condicion dominante.

\section{Risks \& Thesis Breakers}

Los principales riesgos son: tasa larga mas alta, expansion de cap rates, deterioro de disponibilidad laboral en senior housing, desaceleracion de ocupacion, adquisiciones con retornos inferiores al costo de capital, dilucion de equity, dependencia excesiva de operadores y compresion del spread entre RevPOR y ExpPOR. La condicion de invalidacion para una futura tesis Value seria que el precio caiga sin deterioro equivalente en NOI, FFO y balance, creando FCF/FFO yield suficiente para compensar duration.

\section{Portfolio Decision}

La decision es \textbf{no incluir WELL bajo filosofia Value}. El activo puede permanecer en watchlist para Quality o para una entrada oportunista si el mercado castiga REITs por tasas sin deteriorar fundamentos. En el portafolio actual, comprar WELL como Value diluiria la disciplina metodologica: seria pagar un multiple premium por una historia muy buena, no capturar un descuento sobre valor intrinseco.
""")


def well_quality_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Market Cap","USD 151.5bn","FactSet Snapshot, cierre 8-may-2026"),
("SS SHO NOI","22.1\\%","Q1 2026, 14 trimestres +20\\%"),
("S\\&P Rating","A-","Outlook estable"),
("FY26 FFO Guide","USD 6.28","Midpoint elevado 11c"),
("Liquidity","USD 11.1bn","Cash + revolver disponible"),
("Score Quality","74/100","Quality invertible con precio exigente")
])}

Welltower clasifica como \textbf{Quality invertible con peso controlado}. La compania combina activos en micromercados atractivos, demanda demografica estructural, bajo crecimiento de oferta, plataforma de operadores, analitica de capital allocation y acceso a capital institucional. La calidad del negocio se observa en la recuperacion de occupancy, el diferencial entre RevPOR y ExpPOR, el crecimiento de NOI y la mejora de balance. A diferencia de un REIT pasivo, WELL intenta convertir operaciones, datos y relaciones en una ventaja reproducible.

El freno de la recomendacion no es operacional sino de valuacion. La accion cotiza con premium significativo frente a FCF y EBITDA, por lo que no merece peso agresivo aunque el negocio sea superior. La inclusion se justifica si el portafolio necesita exposicion a real estate defensivo con crecimiento secular, pero el sizing debe reconocer duration, tasas y riesgo de adquisiciones. El dictamen final es inclusion selectiva, no posicion core sin restriccion.

\section{Investment View}

Fundada en 1970, Welltower es un REIT de healthcare real estate con foco en seniors housing, wellness housing, triple-net senior housing y outpatient medical. Opera principalmente en Estados Unidos, Canada y Reino Unido. Su posicionamiento se ha movido desde "propietario de inmuebles" hacia una plataforma operativa de capital, datos y alianzas con operadores regionales.

Bajo Quality, el comite debe preguntar si el moat es persistente, si el flujo es durable y si la compania puede reinvertir capital a retornos superiores. WELL tiene una respuesta razonable: la demanda de adultos mayores no depende enteramente del ciclo economico; la oferta nueva sigue limitada; la cartera esta concentrada en micromercados de mayor ingreso; y la compania tiene acceso a multiples fuentes de capital. La combinacion permite capturar alpha si el equipo ejecuta adquisiciones y operaciones con disciplina.

La tesis no es libre de objeciones. El ROIC contable no luce como el de una empresa asset-light, el leverage sigue siendo relevante y el precio actual reduce retorno esperado. Por eso el memo no propone clasificacion de Quality alta conviccion; propone Quality invertible con control de peso y monitoreo de spread operativo.

\section{Business Model \& Cash Engine}

La generacion de flujo proviene de NOI inmobiliario, FFO, adquisiciones, desarrollo, financiamiento eficiente y recuperacion de margen en seniors housing. En SHO, la compania captura directamente ocupacion, precio y eficiencia operativa; en outpatient medical captura renta medica mas estable; en triple-net obtiene flujos contractuales de operadores. Esta mezcla crea balance entre crecimiento operativo y estabilidad de renta.

{table(["Fuente de calidad", "Evidencia", "Por que importa", "Riesgo residual"], [
["Demografia", "80+ poblacion en expansion", "Demanda secular no depende solo de PIB", "Affordability y personal operativo."],
["Ocupacion", "+370 bps en Q1", "Mayor utilizacion apalanca margen", "Normalizacion puede desacelerar."],
["Pricing", "RevPOR +5.5\\% esperado", "Poder de precio superior a expense growth", "Presion laboral y salarios."],
["Capital", "USD 10.5bn inversiones YTD", "Acceso a oportunidades cuando otros no pueden", "Riesgo de pagar precios altos."],
], widths="lXXX")}

El punto de calidad esta en que el crecimiento de NOI no depende solo de adquisiciones. La presentacion de Q1 2026 reporta same-store SHO NOI +22.1\%, revenue same-store +9.5\%, occupancy +370 bps y una expansion de margen de 320 bps. Esa combinacion sugiere poder operativo, no solo apalancamiento financiero.

\section{Competitive Position \& Addressable Market}

WELL compite contra otros REITs, private equity, operadores y capital institucional. Su ventaja se origina en tres planos: acceso a capital, seleccion de micromercados y relacion con operadores. En seniors housing, las dispersiones operativas son amplias; por eso una plataforma con datos, densidad y operadores alineados puede sostener resultados superiores frente a propietarios menos sofisticados.

El TAM se beneficia de la expansion de poblacion 80+, baja construccion nueva y migracion de cuidado hacia entornos mas costo-eficientes. La compania no vende una marca de consumo masivo; vende infraestructura y experiencia operativa a un mercado con demanda estructural. El moat, sin embargo, no es inmune: si el capital vuelve agresivamente al sector, las adquisiciones podrian comprimirse y el spread de reinversion bajaria.

\section{Macro \& Liquidity Backdrop}

La desaceleracion ordenada favorece negocios con demanda no discrecional relativa y acceso a capital. Welltower puede aprovechar un mercado inmobiliario donde compradores menos capitalizados enfrentan fondeo mas caro. Su capacidad de usar equity, deuda unsecured, revolvers, private capital y disposiciones le da flexibilidad en la plomeria financiera.

La misma plomeria crea el principal riesgo: si la curva larga sube o los cap rates se expanden, el valor presente de FFO cae y el costo de capital limita adquisiciones. Quality no ignora precio; simplemente permite pagar un premium cuando la durabilidad es superior. En WELL, el premium es alto, por lo que el portafolio debe exigir confirmacion de FFO per share, no solo crecimiento bruto de activos.

\section{Financial Profile \& Estimate Quality}

FactSet proyecta expansion sustancial de ingresos, EBITDA y FCF hacia 2028. La calidad de estimacion es razonable porque el guidance incorpora adquisiciones anunciadas y porque la trayectoria de occupancy y NOI tiene datos operativos recientes. El punto a vigilar es que EPS GAAP no es la metrica central para un REIT; FFO, NOI, occupancy y leverage explican mejor la calidad economica.

{table(["Metrica", "2023", "2024", "2025", "2026E", "2027E", "2028E"], [
["Ingresos (USD m)", "6,642", "8,119", "13,021", "26,475", "30,958", "37,508"],
["EBITDA (USD m)", "2,518", "2,943", "2,678", "5,306", "6,249", "6,938"],
["FCF (USD m)", "1,183", "1,393", "2,108", "3,835", "4,391", "5,156"],
["Total debt (USD m)", "16,119", "16,758", "21,380", "18,163", "18,086", "--"],
["Total assets (USD m)", "44,012", "51,044", "67,303", "69,224", "71,385", "68,559"],
], widths="lrrrrrr", size="scriptsize")}

El balance se ha fortalecido: Net Debt/EBITDA bajo a 5.3x desde 6.3x y FactSet DCS registra S\\&P A-. Para un REIT con adquisiciones relevantes, esa mejora es parte central de la tesis Quality porque reduce probabilidad de equity issuance forzada y permite financiar oportunidades sin comprometer grado de inversion.

\section{Quality of Earnings, ROIC \& Capital Discipline}

La calidad de earnings debe juzgarse por FFO/NOI mas que por EPS. La compania esta incrementando FFO guidance, generando NOI organico, expandiendo margen SHO y manteniendo acceso a liquidez. No obstante, la rentabilidad contable tradicional es baja por intensidad de activos y depreciacion inmobiliaria. Eso impide tratar a WELL como Quality compounder puro del estilo asset-light.

{score_table([
["Rentabilidad superior", "20", "13", "NOI y FFO fuertes; ROIC contable bajo por estructura REIT."],
["Persistencia y estabilidad", "15", "12", "Demografia y baja oferta favorecen durabilidad."],
["Calidad de flujo", "20", "15", "FFO y FCF mejoran, con yield bajo por precio."],
["Balance y solvencia", "15", "12", "A-, liquidez amplia y leverage en descenso."],
["Moat y posicion competitiva", "10", "8", "Datos, operadores y micromercados crean ventaja."],
["Capital allocation", "10", "8", "Track record activo; riesgo de sobrepagar en ciclo caliente."],
["Valuacion y macro", "10", "6", "Calidad alta, pero precio y tasas limitan conviccion."],
])}

\section{Valuation, Reverse DCF \& Credit}

La valuacion es el principal descuento al score. Un comprador actual esta pagando por la continuidad de crecimiento SHO, expansion de margen, adquisiciones creadoras de valor y baja dislocacion de cap rates. El reverse DCF no exige colapso imposible, pero si exige que el flywheel operativo continue varios anos sin deterioro material. Quality permite pagar por durabilidad; no permite ignorar que el precio ya descuenta parte de ella.

Credito es un punto fuerte relativo. La compania reporta cash e inversiones de corto plazo de USD 4.8bn, lineas disponibles de USD 6.25bn y rating S\\&P A-. Net Debt/EBITDA de 5.3x sigue siendo alto frente a empresas industriales, pero es manejable dentro del contexto REIT y esta mejorando. La tesis de inclusion requiere que esa mejora continue.

\section{Risks \& Thesis Breakers}

Los thesis breakers son: FFO per share estancado pese a adquisiciones, ocupacion SHO debajo de expectativas, RevPOR creciendo por debajo de ExpPOR, deuda/EBITDA volviendo a subir, equity issuance dilutiva, deterioro de operadores o expansion persistente de cap rates. Si dos o mas de estos indicadores se materializan, WELL dejaria de ser Quality invertible y pasaria a Quality trap por precio.

\section{Portfolio Decision}

La recomendacion es \textbf{inclusion moderada bajo Quality}. WELL ofrece exposicion a un tema estructural con mejor ejecucion que el REIT promedio, pero no debe competir por el mismo peso que un compounder con alto ROIC, baja deuda y valuacion razonable. La posicion debe ser tactica-estructural: participar del flywheel, pero con disciplina de entrada y monitoreo de FFO per share.
""")


def axp_value_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Market Cap","USD 215.6bn","FactSet Snapshot, cierre 8-may-2026"),
("P/E LTM","19.7x","Razonable para franquicia premium"),
("FCF Yield","6.96\\%","FactSet cash flow Q1 annualizado"),
("ROE","33.8\\%","Rentabilidad estructural alta"),
("Target Upside","15.5\\%","Consenso FactSet"),
("Score Value","72/100","Value invertible condicionado")
])}

American Express clasifica como \textbf{Value invertible condicionado}. La accion no es deep value: el upside al target medio es 15.5\%, menor al margen de seguridad estricto de 20\%. Sin embargo, la franquicia ofrece ROE superior, FCF yield atractivo, crecimiento de ingresos de 9\%-10\% guiado para 2026, credito controlado y un modelo cerrado de pagos que justifica pagar un multiple moderado. La tesis Value no descansa en re-rating agresivo; descansa en comprar una franquicia de alta calidad a un precio que no exige perfeccion.

El comite debe reconocer el trade-off: AXP no es barata frente a bancos ciclicos, pero si parece razonable frente a redes de pago y franquicias premium. A P/E LTM de 19.7x y EV/EBITDA LTM de 9.3x, el mercado no esta regalando el activo, pero tampoco capitaliza plenamente su capacidad de crecer EPS, fees y spend en una base de clientes premium. La inclusion es defendible con peso medio y condicionada a que la calidad crediticia se mantenga por debajo de niveles de estres.

\section{Investment View}

American Express fue fundada en 1850 y cotiza en NYSE bajo el ticker AXP. La compania combina emision de tarjetas, red de pagos, acquiring de comercios, servicios a consumidores premium, pequenas empresas y corporativos. Su marca se asocia con confianza, servicio, seguridad, experiencias premium y membresia. A diferencia de redes puras, AXP captura merchant discount, card fees, net interest income y datos propios por su modelo cerrado.

La tesis Value parte de una observacion: el negocio tiene caracteristicas de Quality, pero el precio no esta tan exigente como el de otros activos de pagos. El retorno esperado proviene de EPS compounding, recompras, crecimiento de fee base y resistencia del cliente premium. El riesgo es que una desaceleracion del consumo, deterioro de credito o presion regulatoria sobre merchant discount comprima el multiple.

\section{Business Model \& Cash Engine}

AXP genera valor a traves de cuatro motores: discount revenue por volumen de gasto, net card fees por membresia y productos premium, net interest income por saldos revolventes y servicios/red de comercios. El modelo cerrado permite observar informacion de tarjeta-habiente y comercio, lo que mejora underwriting, marketing, rewards y negociacion con merchants. Esa data advantage es una fuente importante de moat.

{table(["Linea de valor", "Evidencia Q1 26", "Lectura Value", "Riesgo"], [
["Revenue growth", "11\\%", "Crecimiento superior para multiple razonable", "Ciclo de consumo."],
["EPS", "USD 4.28, +18\\%", "Apalancamiento operativo y recompra", "Provisionamiento."],
["Net card fees", "+18\\%", "Mayor recurrencia y premium mix", "Fatiga de cuotas o competencia."],
["Write-off rate", "2.3\\%", "Credito aun controlado", "Normalizacion laboral negativa."],
], widths="lXXX")}

El cash engine es atractivo para Value porque combina fees recurrentes y gasto transaccional con credito selectivo. La base premium reduce volatilidad relativa frente a emisores subprime, aunque no elimina riesgo. En Q1 2026, management reporto crecimiento de revenue 11\%, EPS +18\%, card member spending +10\% y reafirmo guidance anual de revenue +9\%-10\% y EPS USD 17.30-17.90.

\section{Competitive Position \& Addressable Market}

AXP compite con Visa, Mastercard, bancos emisores, fintechs, wallets, travel platforms y programas de lealtad. Su ventaja no es escala absoluta de red frente a Visa/Mastercard; es una base premium, mayor intensidad de gasto, experiencias, closed-loop data y capacidad de cobrar fees por membresia. Mas de 100 millones de ubicaciones comerciales dentro de su red amplian aceptacion, mientras la marca sostiene aspiracionalidad.

El mercado direccionable se expande por gasto de consumidores premium, pequenas empresas, travel and entertainment, pagos comerciales, servicios financieros y monetizacion de datos. La oportunidad de Millennial y Gen Z es relevante: management reporto que mas de 70\% de nuevas cuentas globales estan en productos con fee y que ese cohort sigue creciendo en gasto. Para Value, esto aporta crecimiento organico sin depender de multiples mas altos.

\section{Macro \& Liquidity Backdrop}

En desaceleracion ordenada, AXP es sensible a gasto discrecional premium, empleo, credit losses y tasas. La narrativa economica tradicional es razonablemente favorable si el empleo se mantiene y el consumidor premium conserva ingreso. La plomeria financiera exige observar funding, securitizaciones, CDS, liquidez y spreads de credito. FactSet DCS reporta CDS de 5 anos en 38.4 bps y S\\&P A-, lo que indica acceso a fondeo sano.

AXP puede beneficiarse de tasas elevadas via NII, pero tasas demasiado altas incrementan costo de fondeo y delinquencies. La clave es que el crecimiento de NII no sea comprado con deterioro de calidad crediticia. En Q1, management senalo que write-off dollars subieron solo 4\% YoY mientras NII crecia doble digito. Esa asimetria favorece la tesis Value mientras se sostenga.

\section{Financial Profile \& Estimate Quality}

Las estimaciones muestran crecimiento ordenado: revenue de USD 72.2bn en 2025 hacia USD 93.4bn en 2028E, EPS de USD 15.38 a USD 23.03 y net income de USD 10.8bn a USD 14.6bn. La calidad de estimacion es alta por cobertura amplia, guidance explicito y visibilidad en fees, spend y saldos.

{table(["Metrica", "2025", "2026E", "2027E", "2028E", "Lectura"], [
["Ingresos (USD m)", "72,229", "79,352", "86,359", "93,411", "CAGR alto de un digito."],
["EBIT (USD m)", "21,957", "21,340", "23,550", "24,860", "Rentabilidad elevada."],
["Net income (USD m)", "10,759", "11,968", "13,304", "14,616", "EPS compounding."],
["EPS diluido", "15.38", "17.61", "20.14", "23.03", "Crecimiento de doble digito."],
["ROE", "33.8\\%", "33.3\\%", "34.0\\%", "35.8\\%", "Franquicia premium."],
], widths="lrrrrX", size="scriptsize")}

El balance es de una financiera, no de una industrial. Por ello, deuda/equity luce alta, pero debe leerse junto con liquidez, activos financieros, reservas, net charge-offs y rating. Total debt/EBITDA de 2.4x, EBITDA/interest de 8.4x y cash/ST investments de USD 53.8bn sostienen la tesis de solvencia.

\section{Margin of Safety \& Value Discipline}

El margen de seguridad es moderado, no amplio. El upside al target medio de FactSet es 15.5\%, por debajo del filtro estricto. Lo que permite clasificar AXP como Value invertible condicionado es la combinacion de FCF yield, ROE, franquicia premium y multiple razonable frente a calidad. Esta no es una tesis de activo castigado; es una tesis de calidad comprada a precio justo con retorno esperado suficiente.

{score_table([
["Margen de seguridad", "20", "12", "Upside de 15.5\\%; insuficiente para alta conviccion Value."],
["Calidad de flujo", "15", "13", "FCF yield cercano a 7\\% y fees crecientes."],
["Rentabilidad y ROE", "10", "9", "ROE arriba de 33\\% de forma persistente."],
["Balance y solvencia", "15", "11", "A-, liquidez amplia y cobertura de intereses solida."],
["Catalizadores", "10", "8", "EPS +18\\%, guidance reafirmado, fees y Gen Z/Millennial."],
["Riesgo de value trap", "15", "10", "No hay deterioro visible, pero credito es el trigger."],
["Macro y liquidez", "15", "9", "Consumidor premium resistente; sensibilidad a ciclo."],
])}

\section{Valuation, Reverse DCF \& Credit}

El reverse DCF cualitativo exige que AXP mantenga crecimiento de revenue cercano a guidance, ROE alto, provisionamiento controlado y recompra disciplinada. No exige un rerating extremo: con EPS creciendo hacia USD 23.03 en 2028E, un multiple terminal razonable puede sostener retorno aceptable. La tesis se rompe si el mercado deja de pagar por premium credit o si delinquencies suben mas rapido que NII.

Credito es parte central del caso. S\\&P A-, CDS 5Y de 38.4 bps, EBITDA/interest de 8.4x y net debt/EBITDA de 0.4x reducen riesgo de funding. La parte que exige vigilancia es Card balances y Other loans: crecieron 8\% YoY a USD 224.2bn, con write-off rate de 2.3\%. Una aceleracion de perdidas por encima de crecimiento de ingresos invalidaria la tesis.

\section{Risks \& Thesis Breakers}

Los riesgos clave son deterioro de empleo premium, caida de travel and entertainment, presion regulatoria sobre merchant discount, competencia de bancos y wallets, deterioro de cobrand partnerships, aumento de rewards cost y normalizacion crediticia mas rapida. Los thesis breakers son net write-off rate sostenidamente por encima de 2019, revenue growth debajo de 7\%, EPS guidance recortado o CDS widening que indique estres de fondeo.

\section{Portfolio Decision}

La recomendacion es \textbf{incluir AXP bajo Value con peso medio y condicion crediticia}. No es deep value, pero ofrece retorno esperado defendible por calidad de franquicia, FCF, ROE y multiple razonable. Debe competir en portafolio contra bancos de alta calidad y redes de pago, no contra cyclicals baratos con menor moat.
""")


def axp_quality_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("ROE","33.8\\%","FactSet Snapshot FY25/LTM"),
("Revenue Q1","11\\%","FX-adjusted 10\\%"),
("EPS Q1","USD 4.28","18\\% YoY"),
("Net Card Fees","+18\\%","Q1 2026 10-Q"),
("S\\&P Rating","A-","Outlook estable"),
("Score Quality","86/100","Quality alta conviccion")
])}

American Express clasifica como \textbf{Quality alta conviccion}. La compania combina marca premium, closed-loop data, base de clientes de alto gasto, fees recurrentes, red global de comercios, underwriting selectivo y rentabilidad sobre capital consistentemente superior. Q1 2026 mostro revenue +11\%, EPS +18\%, card member spending +10\%, net card fees +18\% y credito controlado, con delinquencies y write-off rates aun por debajo de 2019 segun management.

La recomendacion es inclusion como activo Quality core dentro de servicios financieros. El principal limite no es el negocio sino el ciclo de credito: AXP es una franquicia excelente, pero no inmune a desempleo, gasto discrecional y fondeo. La posicion debe sostenerse mientras la compania conserve ROE superior, crecimiento de fee base, discipline de provisionamiento y fortaleza de marca en consumidores premium jovenes.

\section{Investment View}

Fundada en 1850 y con sede en Nueva York, American Express opera una plataforma global de pagos y premium lifestyle. Sus productos incluyen tarjetas de consumo, small business, corporate cards, merchant acquiring, red de pagos, travel, experiencias, rewards, depositos y lending. La marca American Express es un activo economico: permite cobrar membresia, atraer consumidores de mayor gasto y negociar valor con comercios.

Bajo Quality, AXP destaca porque sus ventajas se refuerzan entre si. El closed-loop network genera datos de ambos lados de la transaccion; esos datos mejoran underwriting, marketing y valor para merchants; la calidad del cliente reduce perdidas; y el producto premium sostiene card fees. No es solo un banco ni una red; es una franquicia de pagos con economics de membresia y credito selectivo.

\section{Business Model \& Cash Engine}

La compania monetiza gasto, membresia y credito. Discount revenue es la mayor linea; net card fees aporta recurrencia y mix premium; NII monetiza saldos; service fees y otros ingresos complementan la red. AXP puede reinvertir sobre-delivery de EPS en marketing, tecnologia, beneficios y adquisicion de card members, manteniendo el ciclo de crecimiento.

{table(["Motor", "Q1 2026", "Calidad economica", "Pregunta de comite"], [
["Revenue", "+11\\%", "Crecimiento superior y diversificado", "Puede sostener 9-10\\% anual?"],
["Spending", "+10\\%", "Engagement de clientes premium", "Se desacelera T\\&E o retail?"],
["Net card fees", "+18\\%", "Recurrencia y pricing power", "Sigue justificando valor percibido?"],
["Credit", "Write-off 2.3\\%", "Perdidas controladas", "Normalizacion queda ordenada?"],
], widths="lXXX")}

La calidad de flujo se fortalece porque las cuotas de tarjeta y relaciones de membresia son menos transaccionales que el puro lending. Esto reduce dependencia de crecer balance para crecer ingresos. Esa diferencia es central frente a bancos: AXP puede expandir earnings por spend, fees, beneficios y red, no solo por prestamo.

\section{Competitive Position \& Addressable Market}

AXP compite con Visa, Mastercard, JPMorgan Chase, Capital One, Citi, fintechs, wallets y plataformas de loyalty. Su moat esta en premium brand, servicio, recompensas, experiencias, aceptacion global y datos cerrados. Management destaco que la base es de consumidores premium y pequenas empresas premium; esa seleccion es relevante porque el crecimiento de Millennial y Gen Z no viene de volumen masivo indiscriminado, sino de usuarios de mayor valor esperado.

El addressable market incluye gasto de consumo premium, PYMES, viajes, dining, business travel, B2B payments, lending selectivo y servicios de membresia. La compania no necesita ganar todo el mercado de pagos; necesita aumentar share-of-wallet dentro de clientes de alto gasto y ampliar aceptacion. Esa estrategia es mas Quality que Growth agresivo: crecimiento con rentabilidad, no volumen por volumen.

\section{Macro \& Liquidity Backdrop}

La desaceleracion ordenada favorece franquicias financieras con clientes de mayor ingreso, buen fondeo y credito disciplinado. AXP tiene sensibilidad a gasto discrecional, pero su base premium suele resistir mejor que segmentos subprime. La plomeria financiera tambien es favorable: FactSet DCS registra S\\&P A-, CDS 5Y de 38.4 bps, total debt/EBITDA de 2.4x y EBITDA/interest de 8.4x.

La compania no esta exenta de tasas. Tasas altas impulsan NII, pero tambien presionan fondeo y delinquencies. El equilibrio actual es favorable porque write-off dollars crecieron solo 4\% YoY mientras NII crece doble digito. Para Quality, esa combinacion indica underwriting robusto y base de clientes menos vulnerable.

\section{Financial Profile \& Estimate Quality}

La calidad financiera es alta por crecimiento visible, ROE superior y guidance reafirmado. Management reitero para 2026 revenue growth de 9\%-10\% y EPS de USD 17.30-17.90. FactSet proyecta EPS de USD 17.61 en 2026E, USD 20.14 en 2027E y USD 23.03 en 2028E. La estimacion es creible porque Q1 ya mostro revenue por encima de guidance anual y la empresa decidio reinvertir parte del over-delivery.

{table(["Metrica", "2025", "2026E", "2027E", "2028E", "Lectura Quality"], [
["Ingresos (USD m)", "72,229", "79,352", "86,359", "93,411", "Crecimiento estable."],
["Net income (USD m)", "10,759", "11,968", "13,304", "14,616", "Escala de earnings."],
["EPS diluido", "15.38", "17.61", "20.14", "23.03", "Compounding visible."],
["ROE", "33.8\\%", "33.3\\%", "34.0\\%", "35.8\\%", "Rentabilidad superior."],
["Debt/Equity", "172.6\\%", "181.1\\%", "174.0\\%", "162.1\\%", "Estructura financiera monitoreable."],
], widths="lrrrrX", size="scriptsize")}

\section{Quality of Earnings, ROIC \& Capital Discipline}

La calidad de earnings es superior por tres razones. Primero, net card fees y membership economics reducen dependencia del credito. Segundo, la seleccion premium limita perdidas relativas. Tercero, el closed-loop data permite mejorar originacion, riesgo y marketing. Las recompras y dividendos pueden ampliar retorno por accion siempre que se mantenga capital regulatorio.

{score_table([
["Rentabilidad superior", "20", "19", "ROE sobre 33\\% con franquicia premium."],
["Persistencia y estabilidad", "15", "13", "Fees, marca y clientes premium sostienen recurrencia."],
["Calidad de flujo", "20", "17", "FCF fuerte y revenue mix de fees mas transacciones."],
["Balance y solvencia", "15", "12", "A-, CDS bajo y cobertura de intereses amplia."],
["Moat y posicion competitiva", "10", "9", "Closed-loop data, marca, servicio y red global."],
["Capital allocation", "10", "8", "Reinversion, marketing y recompras disciplinadas."],
["Valuacion y macro", "10", "8", "Precio razonable, aunque ciclico."],
])}

\section{Valuation, Reverse DCF \& Credit}

La valuacion no es barata absoluta, pero es razonable para Quality. P/E LTM de 19.7x, EV/EBITDA de 9.3x y upside de consenso de 15.5\% no exigen una historia imposible. El reverse DCF cualitativo requiere crecimiento high-single digit de ingresos, ROE alto y credito normalizado. Esas condiciones son consistentes con guidance y con la posicion competitiva actual.

El credito es el termometro central. Total Card balances and Other loans fueron USD 224.2bn, +8\%; net write-off rate fue 2.3\%; y management reporto delinquencies/write-offs debajo de 2019. Si esos indicadores se deterioran mas rapido que revenue o NII, el caso Quality se debilita. Por ahora, la evidencia favorece resiliencia.

\section{Risks \& Thesis Breakers}

Riesgos: deterioro del consumidor premium, competencia en rewards, regulacion de interchange/merchant discount, perdida de cobrand partners, ciberseguridad, funding, aumento de reservas y presion en travel. Thesis breakers: revenue growth debajo de guidance, write-off rate por encima de niveles normalizados sin compensacion de NII, perdida de share en consumidores jovenes premium o caida de ROE debajo de 25\% sostenida.

\section{Portfolio Decision}

La decision es \textbf{incluir AXP como Quality core}. La franquicia tiene moat, rentabilidad, crecimiento y balance suficientemente robustos para justificar inclusion. El peso debe ser mayor que un banco ciclico promedio y menor que una red pure-play sin riesgo de credito directo. En comite, AXP debe defenderse como compounder financiero premium, no como prestamista barato.
""")


def wdc_growth_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Market Cap","USD 165.5bn","FactSet Snapshot, cierre 8-may-2026"),
("Revenue Q3 FY26","+45\\% YoY","Earnings call, 30-abr-2026"),
("Cloud Mix","89\\%","USD 3.0bn, +48\\% YoY"),
("Gross Margin","50.5\\%","Q3 FY26, guia 51-52\\%"),
("FCF Margin","29\\%","USD 978m Q3 FY26"),
("Score Growth","78/100","Growth invertible, precio vigilado")
])}

Western Digital clasifica como \textbf{Growth invertible con peso controlado}. Tras la separacion de SanDisk, WDC es una compania enfocada en HDD, posicionada como proveedor de almacenamiento de alta capacidad para hyperscalers, cloud y workloads de AI. El crecimiento actual es fuerte: revenue +45\% YoY, cloud +48\%, gross margin 50.5\%, FCF margin 29\% y guia Q4 de revenue USD 3.65bn, +40\% YoY en el punto medio. La tesis Growth es real.

La advertencia es la valuacion y el grado de ciclicidad. La accion ya refleja gran parte de la narrativa AI-storage: +178.6\% YTD y +988.4\% 1Y segun FactSet Snapshot, con implied return al target medio de solo 4.8\%. Por eso la inclusion debe ser tactica, no ciega. WDC merece capital si el comite acepta que la demanda de nearline HDD y UltraSMR prolongara el ciclo de crecimiento; no merece peso agresivo si el objetivo es margen de seguridad.

\section{Investment View}

Western Digital fue fundada en 1970 y cotiza en NASDAQ bajo el ticker WDC. Luego de la separacion de su negocio flash/SanDisk en febrero de 2025, la compania opera como pure-play HDD. Sus productos atienden cloud data centers, enterprise, client y consumer, con mayor exposicion economica a nearline drives de alta capacidad para hyperscalers. Las marcas centrales son Western Digital y WD; SanDisk quedo como compania separada.

La tesis Growth se basa en que AI no solo requiere compute; tambien requiere almacenar datos, checkpoints, logs, synthetic data, datasets fisicos, observabilidad y cold/warm data. Management describio tres drivers: crecimiento de datos tradicional, agentic AI/inference y physical AI/synthetic data. Si estos drivers persisten, HDD de alta capacidad sigue siendo una infraestructura costo-eficiente y WDC puede capturar pricing, mix y margen.

\section{Business Model \& Cash Engine}

WDC genera flujo vendiendo dispositivos y soluciones HDD, principalmente a cloud y hyperscale data centers. El motor de valor actual no es volumen de unidades tradicional, sino mix hacia mayores capacidades, UltraSMR, pricing strategy, eficiencia de costos y menor necesidad de capex unitario. Management senalo que no planea incrementar capacidad de unidades, sino mejorar areal density y capacidad por drive.

{table(["Unidad Q3 FY26", "Ingresos", "\\% total", "Crec. YoY"], [
["Cloud", "USD 2,972m", "89\\%", "48\\%"],
["Consumer", "USD 186m", "6\\%", "24\\%"],
["Client", "USD 179m", "5\\%", "31\\%"],
["Total", "USD 3,337m", "100\\%", "45\\%"],
], widths="lrrr")}

La calidad del crecimiento mejora porque el margen acompana al revenue. Gross margin llego a 50.5\% y la guia Q4 apunta a 51\%-52\%. Operating cash flow fue USD 1.1bn y FCF USD 978m, equivalente a 29\% de margen. Este no es Growth sin caja; es crecimiento con fuerte conversion de flujo, aunque parte del resultado puede estar favorecido por ciclo de demanda y capacidad restringida.

\section{Competitive Position \& Addressable Market}

El mercado competitivo incluye Seagate, proveedores de almacenamiento enterprise, arquitecturas flash y cambios en infraestructura cloud. La ventaja de WDC reside en tecnologia HDD de alta capacidad, relaciones con hyperscalers, roadmap de ePMR/HAMR/UltraSMR y capacidad para entregar menor costo por terabyte. La adopcion UltraSMR es clave: management espera tener a sus principales clientes calificados hacia finales de 2027 y que cerca de 60\% de exabytes enviados en FY27 utilicen UltraSMR.

El TAM se expande por cloud, AI inference, agentic workflows, synthetic data, video, observabilidad, backups y archives. HDD no compite por todos los workloads de AI; compite por data lake, almacenamiento de escala y costo por TB. Esta distincion es importante para el comite: WDC no es una semiconductor AI beta puro, sino una infraestructura de datos donde la demanda crece por los residuos y activos informacionales que AI produce.

\section{Macro \& Liquidity Backdrop}

La desaceleracion ordenada puede afectar capex empresarial general, pero hyperscale AI/cloud tiene un ciclo propio de inversion. La plomeria financiera es favorable para WDC porque el balance se desendeudo tras la separacion: FactSet DCS muestra S\\&P BBB-, total debt/EBITDA 0.4x, EBITDA/interest 17.9x y liquidez total de USD 4.49bn. La reduccion de deuda por el intercambio con SanDisk disminuye riesgo de balance y aumenta optionality de capital return.

El riesgo macro de Growth esta en tasas y liquidez de equity. WDC tiene beta 1.77 y el mercado ya la trata como duration/cycle equity ligada a AI. Si se contrae apetito por riesgo, el multiple puede comprimirse aunque fundamentals sigan buenos. La tesis debe estar soportada por earnings y FCF, no solo por narrativa de AI.

\section{Financial Profile \& Estimate Quality}

Las cifras proyectadas son extraordinarias, pero deben leerse con cautela por la separacion de SanDisk y la cyclicality del negocio. FactSet proyecta ingresos desde USD 9.5bn en 2025 hacia USD 22.3bn en 2028E, EBITDA de USD 2.6bn a USD 12.7bn y FCF de USD 1.45bn a USD 7.92bn. La estimacion tiene soporte en Q3/Q4 guidance, pero el rango de outcomes es amplio.

{table(["Metrica", "2025", "2026E", "2027E", "2028E", "Lectura Growth"], [
["Ingresos (USD m)", "9,519", "12,845", "17,562", "22,323", "Crecimiento AI/cloud."],
["EBITDA (USD m)", "2,613", "4,997", "8,612", "12,737", "Fuerte leverage operativo."],
["EPS diluido", "4.82", "20.18", "17.32", "24.50", "Volatilidad por separacion/ciclo."],
["FCF (USD m)", "1,454", "3,469", "5,757", "7,915", "Caja elevada."],
["Gross margin", "40.4\\%", "48.0\\%", "54.6\\%", "56.3\\%", "Mix y pricing."],
], widths="lrrrrX", size="scriptsize")}

La calidad de estimacion depende de tres variables: duracion de la demanda hyperscale, disciplina de oferta en la industria HDD y capacidad de sostener pricing con mayor densidad. Si las tres se mantienen, WDC puede transformar un ciclo en un nuevo nivel de margen. Si una falla, el multiple actual se vuelve vulnerable.

\section{Quality of Growth, ROIC \& Reinvestment}

WDC cumple bien con crecimiento, TAM y escalabilidad de margen; cumple parcialmente con recurrencia. Los ingresos dependen de ciclos de capex de pocos hyperscalers: el 10-Q indica que los top 10 clientes representaron 76\% de revenue trimestral. Esa concentracion no invalida Growth, pero exige descuento en score por customer risk.

{score_table([
["Crecimiento de ingresos", "15", "14", "Q3 +45\\% y guia Q4 +40\\%."],
["Calidad del crecimiento", "15", "12", "Margen y FCF acompanan, pero ciclo puede normalizar."],
["Mercado direccionable", "10", "9", "AI, cloud, inference y synthetic data amplian demanda."],
["Ventaja competitiva", "10", "8", "UltraSMR, roadmap y relaciones hyperscale."],
["Escalabilidad y margenes", "10", "9", "Gross margin 50.5\\%, guia 51-52\\%."],
["ROIC y reinversion", "15", "11", "ROIC mejora; historial post-spin aun corto."],
["Balance y financiamiento", "10", "9", "Deleveraging fuerte y liquidez alta."],
["Valuacion y expectativas", "10", "3", "Precio incorpora gran parte del upside."],
["Macro-liquidez", "5", "3", "Beta elevada y sensibilidad a risk appetite."],
])}

\section{Valuation, Reverse DCF \& Credit}

La valuacion es el principal punto de tension. FactSet Snapshot muestra market cap de USD 165.5bn, EV de USD 189.7bn, P/E LTM de 28.7x, EV/EBITDA LTM de 47.0x y target upside de 4.8\%. El reverse DCF cualitativo exige que WDC sostenga margenes superiores, FCF elevado y crecimiento de cloud por varios anos. La tesis puede funcionar, pero no tolera un rollover rapido del ciclo.

Credito es una fortaleza emergente. S\\&P BBB-, deuda total de USD 1.6bn, cash de USD 3.2bn y total debt/EBITDA de 0.4x reducen riesgo financiero. La compania tambien menciono dividendos y devolucion de exceso de FCF a accionistas. Para Growth, esto reduce el riesgo de financiacion y permite que el upside dependa mas de ejecucion operativa que de balance.

\section{Risks \& Thesis Breakers}

Riesgos: concentracion hyperscaler, retraso en UltraSMR/HAMR, sustitucion por flash en ciertos workloads, sobreoferta HDD, pricing downcycle, capex cloud menor, ejecucion post-spin, multiples excesivos y volatilidad por beta. Thesis breakers: gross margin debajo de 45\%, guia de revenue recortada, cloud mix perdiendo crecimiento, FCF margin bajo 15\%, o evidencia de que la demanda AI era pull-forward y no estructural.

\section{Portfolio Decision}

La recomendacion es \textbf{incluir WDC bajo Growth con peso tactico-controlado}. El activo tiene crecimiento, caja y balance, pero la accion ya desconto mucho. En portafolio, WDC debe tratarse como exposicion a infraestructura de datos AI con riesgo ciclico, no como compounder defensivo. La posicion se justifica mientras revenue cloud, margen bruto y FCF sigan superando expectativas.
""")


def wdc_quality_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Gross Margin","50.5\\%","Q3 FY26"),
("FCF Margin","29\\%","USD 978m Q3 FY26"),
("Total Debt/EBITDA","0.4x","FactSet DCS"),
("S\\&P Rating","BBB-","Outlook estable"),
("Customer Top 10","76\\% revenue","Concentracion relevante"),
("Score Quality","66/100","Quality Watchlist")
])}

Western Digital no clasifica todavia como Quality de alta conviccion; clasifica como \textbf{Quality Watchlist}. La empresa ha mejorado mucho: es un pure-play HDD tras SanDisk, tiene margen bruto superior a 50\%, FCF fuerte, balance desendeudado y posicion estrategica en cloud/AI storage. Pero la metodologia Quality exige persistencia a traves del ciclo, estabilidad de moat, recurrencia y retorno sobre capital probado. En WDC, la evidencia post-spin es prometedora, aunque todavia corta y ciclica.

La recomendacion es no usar WDC como posicion Quality core. Puede estar en portafolio bajo Growth, pero bajo Quality debe esperar mas evidencia: margenes sostenidos, menor concentracion de clientes, ROIC estable sobre WACC y resiliencia ante normalizacion de demanda. El riesgo no es insolvencia; el riesgo es confundir un excelente tramo de ciclo con calidad estructural permanente.

\section{Investment View}

Western Digital fue fundada en 1970 y actualmente opera como compania enfocada en HDD despues de separar el negocio flash/SanDisk en febrero de 2025. Sus productos sirven a cloud, client y consumer, pero la economia actual depende de cloud/hyperscale. El posicionamiento es claro: infraestructura de almacenamiento de alta capacidad para la economia de datos impulsada por AI.

Bajo Quality, el analisis debe ser mas exigente que bajo Growth. La pregunta no es si WDC crece hoy, sino si puede sostener retornos superiores cuando la industria pase por digestion de inventarios, pricing pressure o menor capex cloud. La respuesta actual es "probablemente, pero aun no probado".

\section{Business Model \& Cash Engine}

WDC vende HDDs y soluciones de almacenamiento, con una mezcla fuertemente sesgada a cloud. El cash engine actual se beneficia de mayor capacidad por drive, mix hacia nearline, UltraSMR, pricing y disciplina de costos. Q3 FY26 mostro FCF de USD 978m y FCF margin de 29\%, un dato muy fuerte para una compania historicamente ciclica.

{table(["Dimension", "Evidencia", "Lectura Quality", "Riesgo"], [
["Margen bruto", "50.5\\%", "Calidad operacional superior", "Puede normalizar si pricing cae."],
["FCF", "USD 978m", "Conversion de caja fuerte", "Ciclo favorable."],
["Balance", "Debt/EBITDA 0.4x", "Riesgo financiero bajo", "Convertible/devolucion de capital."],
["Concentracion", "Top 10 = 76\\%", "Relaciones estrategicas", "Dependencia de pocos clientes."],
], widths="lXXX")}

\section{Competitive Position \& Addressable Market}

WDC tiene posicion relevante en HDD de alta capacidad, tecnologia UltraSMR y relaciones con hyperscalers. Su ventaja depende de roadmap tecnologico, fiabilidad, costo por TB y capacidad de calificar productos con clientes grandes. Estos elementos son moats industriales, pero no son tan estables como una red de pagos o una marca premium: la industria de hardware puede cambiar por tecnologia, ciclos de oferta y decisiones de capex de clientes concentrados.

El TAM de almacenamiento de datos es amplio y probablemente crece con AI, pero Quality exige que el TAM se traduzca en retornos defensibles. Si WDC captura capacidad con pricing racional, la tesis puede migrar hacia Quality invertible. Si el crecimiento se compra con presion de precio o capex, seguira siendo un activo Growth/cyclical.

\section{Macro \& Liquidity Backdrop}

La plomeria financiera ha mejorado mucho. FactSet DCS muestra S\\&P BBB-, cash de USD 3.2bn, total debt de USD 1.6bn, liquidez total de USD 4.49bn, total debt/EBITDA de 0.4x y EBITDA/interest de 17.9x. El balance ya no es el cuello de botella. Esto eleva la calidad de la tesis respecto a ciclos anteriores.

El entorno macro sigue siendo sensible para hardware: si la liquidez se endurece o hyperscalers moderan capex, la demanda puede caer rapido. Ademas, la accion tiene beta 1.77 y una revalorizacion muy fuerte. Quality no debe depender de momentum bursatil; debe depender de resiliencia operativa. Aun falta observar esa resiliencia en un ciclo adverso post-separacion.

\section{Financial Profile \& Estimate Quality}

Las estimaciones muestran una compania transformada: EBITDA de USD 2.6bn en 2025 a USD 12.7bn en 2028E, FCF de USD 1.45bn a USD 7.92bn y margenes brutos hacia 56.3\%. Esta trayectoria es atractiva, pero la calidad de estimacion es media, no alta, porque el negocio acaba de cambiar estructura corporativa y el ciclo HDD esta en fase muy favorable.

{table(["Metrica", "2025", "2026E", "2027E", "2028E", "Lectura Quality"], [
["Ingresos (USD m)", "9,519", "12,845", "17,562", "22,323", "Crecimiento fuerte, aun ciclico."],
["EBITDA (USD m)", "2,613", "4,997", "8,612", "12,737", "Escala y mix favorable."],
["FCF (USD m)", "1,454", "3,469", "5,757", "7,915", "Caja solida."],
["ROE", "20.0\\%", "35.1\\%", "44.0\\%", "46.1\\%", "Muy alto, pero post-spin."],
["Debt/Equity", "95.7\\%", "19.0\\%", "13.2\\%", "8.9\\%", "Balance mejora rapido."],
], widths="lrrrrX", size="scriptsize")}

\section{Quality of Earnings, ROIC \& Capital Discipline}

El mayor avance de calidad es la conversion de flujo y la reparacion del balance. El mayor descuento es la persistencia. Q3 FY26 contiene elementos muy fuertes, pero algunos datos pueden estar afectados por separacion, costos no recurrentes, deuda-for-equity exchange y fase favorable de pricing. Una Quality thesis necesita ver continuidad por varios trimestres sin apoyo de sorpresa ciclica.

{score_table([
["Rentabilidad superior", "20", "13", "Margen y ROE altos, historial post-spin corto."],
["Persistencia y estabilidad", "15", "7", "Demanda cloud fuerte, pero hardware ciclico."],
["Calidad de flujo", "20", "15", "FCF margin 29\\%, balance liquido."],
["Balance y solvencia", "15", "14", "Debt/EBITDA 0.4x y liquidez robusta."],
["Moat y posicion competitiva", "10", "7", "UltraSMR y clientes hyperscale; concentracion alta."],
["Capital allocation", "10", "6", "Deleveraging positivo; capital return aun por probar."],
["Valuacion y macro", "10", "4", "Precio exigente y beta alta."],
])}

\section{Valuation, Reverse DCF \& Credit}

La valuacion impide una recomendacion Quality core. P/E LTM de 28.7x y EV/EBITDA LTM de 47.0x implican que el mercado ya anticipa mejora importante. Quality puede aceptar premiums cuando la durabilidad esta probada; WDC todavia no ha demostrado esa estabilidad en todo el ciclo HDD post-spin. El reverse DCF exige continuidad de gross margin arriba de 50\% y FCF robusto, condiciones posibles pero exigentes.

Credito, en cambio, es favorable. La reduccion de deuda por la separacion de SanDisk y la liquidez total de USD 4.49bn disminuyen riesgo de balance. La compania puede financiar operaciones, capex disciplinado y retornos de capital con FCF. Esta fortaleza evita una clasificacion negativa; simplemente no basta para elevarla a Quality alta conviccion.

\section{Risks \& Thesis Breakers}

Riesgos: concentracion de clientes, cambio tecnologico, exceso de oferta, pricing pressure, retraso en roadmap, demanda AI menor, dependencia de hyperscalers, multiple elevado y volatilidad de beta. Thesis breakers para Quality: gross margin normalizando por debajo de 40\%, FCF margin menor a 10\%, deuda volviendo a subir, perdida de clientes clave o recorte de guidance que muestre que Q3 fue pico de ciclo.

\section{Portfolio Decision}

La decision es \textbf{no incluir WDC como Quality core; mantener en Watchlist Quality}. El activo puede comprarse bajo Growth si se acepta riesgo de ciclo y valuacion, pero Quality requiere mas evidencia de persistencia. Para migrar a Quality invertible, WDC debe sostener ROIC/FCF alto por varios trimestres, reducir dependencia de pocos clientes y demostrar que el margen actual no es solo una fase de escasez.
""")


memos = [
    {**well_common, "philosophy": "Value", "preview": "No inclusion Value: negocio fuerte, precio sin margen de seguridad.", "file": "WELL_value_investment_memo.tex", "body": well_value_body()},
    {**well_common, "philosophy": "Quality", "preview": "Quality invertible: excelente plataforma, valuacion y tasas limitan peso.", "file": "WELL_quality_investment_memo.tex", "body": well_quality_body()},
    {**axp_common, "philosophy": "Value", "preview": "Value condicionado: franquicia premium a precio razonable, no deep value.", "file": "AXP_value_investment_memo.tex", "body": axp_value_body()},
    {**axp_common, "philosophy": "Quality", "preview": "Quality core: marca, ROE, fees y closed-loop data sostienen el moat.", "file": "AXP_quality_investment_memo.tex", "body": axp_quality_body()},
    {**wdc_common, "philosophy": "Growth", "preview": "Growth invertible: AI storage y FCF alto, pero precio ya exige ejecucion.", "file": "WDC_growth_investment_memo.tex", "body": wdc_growth_body()},
    {**wdc_common, "philosophy": "Quality", "preview": "Quality Watchlist: mejora real, persistencia post-spin aun no probada.", "file": "WDC_quality_investment_memo.tex", "body": wdc_quality_body()},
]


for memo in memos:
    render(memo)
    print(f"Wrote {memo['file']}")
