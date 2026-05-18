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
  \node[anchor=west,text=white,font=\huge\bfseries,text width=5.75in,align=left] at ([xshift=0.95in,yshift=-3.15in]current page.north west) {<<COMPANY>>};
  \node[anchor=east,text=white,font=\large] at ([xshift=-0.95in,yshift=-2.70in]current page.north east) {10 de mayo de 2026};
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
    blocks = []
    for i in range(0, len(metrics), 3):
        blocks.append("\\noindent\n" + "\n\\hfill\n".join(
            [rf"\metricbox{{{a}}}{{{b}}}{{{c}}}" for a, b, c in metrics[i:i + 3]]
        ) + "\n\n\\vspace{1.0em}")
    return "\n\n".join(blocks)


def score_table(rows):
    return table(["Dimension metodologica", "Peso", "Score", "Lectura de comite"], rows, "lrrX", "scriptsize")


def interpolate(template):
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
            out.append(str(eval(template[i + 1:j], globals(), locals())))
            i = j + 1
        else:
            out.append(template[i])
            i += 1
    return "".join(out).replace("\\\\&", "\\&").replace("\\\\%", "\\%")


def render(memo):
    tex = PREAMBLE.replace("<<TICKER>>", memo["ticker"]).replace("<<PHILOSOPHY>>", memo["philosophy"])
    tex += TITLE
    for key in ["company", "philosophy", "exchange", "ticker", "sector", "preview"]:
        tex = tex.replace(f"<<{key.upper()}>>", memo[key])
    tex += memo["body"] + "\n\\end{document}\n"
    (OUT / memo["file"]).write_text(tex, encoding="utf-8")
    print(f"Wrote {memo['file']}")


def parr_growth_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Market Cap","USD 3.23bn","FactSet, cierre 8-may-2026"),
("Revenue FY26E","USD 7.98bn","Recuperacion moderada esperada"),
("EPS FY26E","USD 11.20","Consenso FactSet"),
("FCF FY25","USD 356m","Conversion relevante para reinversion"),
("ROIC FY25","13.9\\%","Recuperacion desde -1.3\\% FY24"),
("Score Growth","63/100","Growth Watchlist")
])}

Par Pacific no es un Growth estructural en sentido puro; clasifica como \textbf{Growth Watchlist}. La compania puede mostrar crecimiento tactico de earnings y FCF por recuperacion de margenes de refinacion, optimizacion de activos, puesta en marcha de Hawaii Renewables y recompras, pero el origen del crecimiento sigue siendo ciclico. Bajo metodologia Growth, esto obliga a distinguir expansion de ciclo de crecimiento durable.

La recomendacion es seguimiento activo, no inclusion agresiva. PARR puede aportar convexidad si el entorno de productos refinados se mantiene ajustado y si el FCF se normaliza despues de presiones de working capital. Pero no debe competir con Growth secular: el TAM no crece como software, la recurrencia es baja y el negocio depende de cracks, diferenciales, turnarounds, inventarios, RINs y demanda de combustibles.

\section{Investment View}

PARR cotiza en NYSE y opera una plataforma downstream integrada con refinerias, retail y logistica en Hawaii, Washington, Wyoming y Montana. Su historia corporativa se remonta a una incorporacion en 1984, aunque la plataforma actual se consolido por adquisiciones desde 2012-2013. Esa diferencia importa: no es un gran refinador integrado; es una plataforma regional construida por activos fisicos, supply chain y capital allocation.

El lente Growth busca runway, monetizacion del crecimiento y reinversion con retorno. En PARR, el runway es menos secular y mas operacional: mayor utilization, captura de margen regional, optimizacion logistica, crecimiento en renewable fuels y recompras que aumentan EPS. La tesis puede funcionar si el EBITDA normalizado sube, pero falla si el supuesto de crecimiento es simplemente extrapolar un pico de refinacion.

\section{Business Model \& Cash Engine}

El cash engine combina Refining, Retail y Logistics. Refining aporta la mayor volatilidad y el mayor potencial de upside; Retail agrega margen de distribucion y cercania al cliente final; Logistics funciona como infraestructura de movimiento, almacenamiento y abastecimiento en mercados donde la geografia crea fricciones. La integracion permite que PARR capture valor en varios puntos de la cadena, pero no elimina el componente commodity.

{table(["Motor", "Escala / dato", "Lectura Growth", "Riesgo"], [
["Refining", "219 Mbpd capacidad", "Mayor throughput y margen pueden acelerar EBITDA", "Cracks y paros operativos."],
["Logistics", "13mm bbl almacenamiento", "Infraestructura mejora captura regional", "Retornos limitados si volumen cae."],
["Retail", "Hele, nomnom, 76", "Margen mas estable y contacto final", "Escala menor frente a grandes redes."],
["Hawaii Renewables", "Proyecto de transicion", "Optionality de crecimiento", "Capex, ejecucion y regulacion."],
])}

\section{Competitive Position \& Addressable Market}

PARR compite de forma distinta a Valero, Marathon o Phillips 66. Su ventaja no es escala nacional, sino posicionamiento local en mercados complejos: Hawaii por geografia insular, Pacific Northwest por logistica y Rockies por acceso/regiones con restricciones. Esto puede sostener margenes cuando el mercado esta ajustado, pero no constituye un moat secular inmune al ciclo.

El TAM relevante es la demanda regional de combustibles, distribucion y almacenamiento, no el mercado global de energia. Bajo Growth, esto limita el score: hay oportunidades de optimizacion y proyectos renovables, pero no hay evidencia suficiente de un mercado expansivo de alta duracion comparable con e-commerce, pagos digitales o AI infrastructure.

\section{Macro \& Liquidity Backdrop}

La desaceleracion ordenada es ambigua para PARR. Si la demanda de combustibles se mantiene y la oferta de productos refinados sigue ajustada, la compania puede capturar margenes elevados. Si el consumo se debilita o se normalizan cracks, el crecimiento se evapora rapido. La plomeria financiera tambien pesa: inventarios y capital de trabajo pueden consumir caja cuando suben precios de crudo, incluso con EBITDA positivo.

Para Growth, el entorno no es un viento de cola estructural; es una opcion ciclica. La liquidez de la compania, cercana a USD 938m, le permite atravesar turnarounds y recomprar acciones, pero el comite debe exigir evidencia de FCF recurrente antes de elevar la posicion.

\section{Financial Profile \& Estimate Quality}

El perfil financiero muestra recuperacion, no estabilidad. FY24 fue debil, FY25 recupero FCF y ROIC, y el consenso FY26E anticipa EPS de USD 11.20. La calidad de estimacion es media porque depende de margenes de refinacion, utilization y capital de trabajo.

{table(["Metrica", "FY23", "FY24", "FY25", "FY26E", "Lectura Growth"], [
["Revenue (USD m)", "8,232", "7,974", "7,465", "7,975", "Recuperacion, no hipercrecimiento."],
["Gross margin", "10.1\\%", "1.9\\%", "7.0\\%", "--", "Alta sensibilidad a ciclo."],
["FCF (USD m)", "588.5", "1.7", "356.4", "--", "Caja atractiva pero no lineal."],
["ROIC", "39.2\\%", "-1.3\\%", "13.9\\%", "--", "Retorno ciclico."],
["EPS GAAP", "9.65", "-0.59", "7.16", "11.20", "Crecimiento por recuperacion."],
], "lrrrrX", "scriptsize")}

\section{Quality of Growth, ROIC \& Reinvestment}

El crecimiento de PARR tiene calidad media-baja bajo una filosofia Growth: puede ser rentable, pero no es recurrente. La mejor defensa es capital allocation. La administracion ha recomprado mas de 14 millones de acciones desde el inicio del programa, mas de 20\% del float, y Q1 2026 incluyo recompra a precio promedio de USD 37.96. Si las recompras ocurren con descuento, pueden transformar FCF ciclico en crecimiento de valor por accion.

{score_table([
["Crecimiento de ingresos", "15", "6", "FY26E recupera ventas, pero sin runway secular claro."],
["Calidad del crecimiento", "15", "7", "FCF puede ser alto; depende de ciclo y working capital."],
["Mercado direccionable", "10", "5", "Mercados regionales defendibles, TAM maduro."],
["Ventaja competitiva", "10", "7", "Activos locales y logistica crean barreras fisicas."],
["Escalabilidad y margenes", "10", "6", "Margen apalanca EBITDA, pero tambien cae rapido."],
["ROIC y reinversion", "15", "9", "ROIC recuperado y buybacks disciplinados."],
["Balance y financiamiento", "10", "8", "Liquidez amplia y leverage manejable."],
["Valuacion y expectativas", "10", "9", "Expectativas no parecen excesivas."],
["Macro-liquidez", "5", "3", "Ciclo energetico y demanda limitan visibilidad."],
])}

\section{Valuation, Reverse DCF \& Credit}

La valuacion ayuda a la tesis Growth porque el mercado no exige crecimiento secular. P/E LTM de 7.2x, EV/EBITDA LTM de 5.3x y FCF yield FY25 cercano a 11\% permiten que un crecimiento modesto de EPS genere retorno. El reverse DCF no exige una historia heroica; exige que EBITDA normalizado no vuelva al escenario FY24 y que el capital allocation siga siendo disciplinado.

Credito es aceptable. La deuda total FY25 fue cercana a USD 803m y la liquidez Q1 2026 rondo USD 938m. No es balance de alta calidad defensiva, pero si es suficiente para una tesis tactica si el ciclo no se deteriora de forma abrupta.

\section{Risks \& Thesis Breakers}

Los riesgos centrales son normalizacion de cracks, CFO negativo por working capital, turnarounds, regulacion ambiental, costos de RINs/CCA, execution risk de Hawaii Renewables y recompras en momentos de sobrevaluacion. La tesis Growth se invalida si FY26 no confirma expansion de EPS/FCF, si EBITDA normalizado cae bajo USD 500m o si el crecimiento depende de deuda y no de caja operativa.

\section{Portfolio Decision}

La decision es \textbf{Watchlist Growth con opcion tactica}. PARR puede entrar si el portafolio busca crecimiento ciclico de caja con valuacion baja y catalizador de recompras. No debe etiquetarse como Growth core porque carece de recurrencia y duracion. El sizing debe ser moderado y ligado a confirmacion de FCF post-turnaround.
""")


def parr_value_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Market Cap","USD 3.23bn","FactSet, cierre 8-may-2026"),
("P/E LTM","7.19x","Baratura estadistica visible"),
("EV/EBITDA LTM","5.33x","Descuento vs refinadores grandes"),
("FCF FY25","USD 356m","FCF yield aprox. 11\\%"),
("Liquidity Q1","USD 938m","Flexibilidad para ciclo"),
("Score Value","69/100","Value condicionado / Watchlist")
])}

Par Pacific presenta una tesis \textbf{Value condicionada}, con sesgo positivo pero sin margen suficiente para alta conviccion. La accion cotiza con multiplos bajos, FCF historicamente atractivo y balance manejable; sin embargo, el negocio es ciclico y el riesgo de value trap es real si el mercado esta capitalizando un EBITDA temporalmente elevado. La compania no debe comprarse solo porque el P/E es bajo; debe comprarse si el comite concluye que el FCF normalizado sostiene el valor intrinseco aun despues de ajustar por cracks, working capital, turnarounds y regulacion ambiental.

El dictamen es \textbf{Watchlist Value activa}. PARR merece cobertura formal y posible inclusion tactica si el portafolio busca exposicion downstream con descuento, recompras y activos regionales. No merece posicion core hasta confirmar que el FCF post-turnaround y el EBITDA normalizado no regresan al escenario debil de FY24.

\section{Investment View}

PARR cotiza en NYSE y opera una plataforma downstream integrada en Hawaii, Washington, Wyoming y Montana. La historia corporativa se remonta a una incorporacion en 1984, aunque la plataforma operativa actual se consolido principalmente por adquisiciones desde 2012-2013. Esta trayectoria importa porque PARR no es un refinador integrado de gran escala; es una coleccion de activos regionales con logistica, retail y refinacion en mercados donde la geografia puede crear barreras.

La tesis Value se apoya en cuatro elementos. Primero, la pantalla de valuacion muestra descuento: P/E LTM de 7.2x, EV/EBITDA LTM de 5.3x y EV/Sales de 0.52x. Segundo, FY25 genero USD 356m de FCF, equivalente a un yield cercano a 11\% sobre market cap. Tercero, el balance no invalida la tesis: liquidez cercana a USD 938m y leverage manejable. Cuarto, la administracion ha usado recompras de forma relevante, con mas de 14 millones de acciones recompradas desde el inicio del programa.

La objecion es igualmente importante: un refinador puede parecer barato justo antes de normalizar margenes. Por eso, la lectura correcta es FCF normalizado, no P/E spot. Si el EBITDA sostenible se acerca a USD 600m, PARR luce razonablemente barato; si cae hacia USD 500m o menos, el descuento se estrecha y la tesis depende mas de timing ciclico que de valor intrinseco.

\section{Business Model \& Cash Engine}

Par Pacific genera flujo a traves de tres segmentos: Refining, Retail y Logistics. Refining procesa crudo y produce gasolina, destilados, asfalto y otros productos; Retail monetiza estaciones y tiendas bajo marcas regionales; Logistics mueve, almacena y distribuye crudo y productos refinados. Esta integracion permite capturar margen en varios puntos de la cadena, pero no elimina la exposicion a commodities.

{table(["Segmento", "Escala / activo", "Contribucion Value", "Riesgo"], [
["Refining", "4 refinerias / 219 Mbpd", "Principal fuente de EBITDA y upside ciclico", "Cracks, turnarounds y utilization."],
["Logistics", "13mm bbl almacenamiento", "Infraestructura regional y captura de margen", "Volumen y tarifas ligados al sistema."],
["Retail", "Hele, nomnom, 76", "Margen mas estable y salida al cliente final", "Menor escala que redes nacionales."],
["Renewables", "Hawaii Renewables", "Optionality y cumplimiento regulatorio", "Capex, ramp-up y retornos inciertos."],
])}

La ventaja de PARR esta en mercados regionales donde la logistica importa. Hawaii tiene complejidad insular; Pacific Northwest y Rockies dependen de activos, acceso y distribucion. Esta estructura puede sostener capturas superiores, pero no crea una franquicia inmune al ciclo. El comite debe exigir que Logistics y Retail reduzcan volatilidad suficiente para justificar valor intrinseco, no solo que acompanen a Refining en un buen ciclo.

\section{Competitive Position \& Addressable Market}

PARR compite contra refinadores grandes, distribuidores, traders, importaciones y alternativas de suministro. Su posicionamiento no esta en escala absoluta, sino en nichos regionales. Frente a Valero, Marathon o Phillips 66, PARR tiene menor liquidez, menor diversificacion y mas volatilidad; frente a esos mismos peers, puede ofrecer mayor asimetria si el mercado subestima la calidad local de sus activos.

{table(["Compania", "Market Cap", "EV", "TEV/EBITDA", "Lectura"], [
["Par Pacific", "3.23bn", "3.08bn", "4.67x", "Nicho regional integrado."],
["Valero", "71.58bn", "60.17bn", "7.84x", "Escala global, mayor liquidez."],
["Marathon Petroleum", "71.49bn", "87.22bn", "8.68x", "Refinacion mas midstream."],
["Phillips 66", "68.78bn", "74.27bn", "10.62x", "Diversificacion amplia."],
["HF Sinclair", "13.06bn", "10.91bn", "5.75x", "Comparable medio cercano."],
], "lrrrX", "scriptsize")}

El TAM relevante es maduro: combustibles, almacenamiento y distribucion en regiones servidas por sus activos. Bajo Value, eso no es problema si el precio descuenta estancamiento y el FCF normalizado es suficiente. El riesgo es que el mercado tenga razon al aplicar descuento por ciclo, regulacion, escala y concentracion regional.

\section{Macro \& Liquidity Backdrop}

El entorno de desaceleracion ordenada crea una lectura mixta. Si la demanda de combustibles se mantiene y la oferta de productos refinados sigue ajustada, PARR puede beneficiarse de margenes elevados. Si el consumo se debilita o se normalizan cracks, la utilidad cae rapido. La plomeria financiera es relevante porque inventarios y capital de trabajo absorben caja cuando los precios de crudo/productos suben, aun si el EBITDA luce sano.

Para Value, la macro no debe ser el pilar de la tesis; debe ser un filtro de supervivencia. PARR tiene liquidez para atravesar volatilidad, pero no tiene la estabilidad de un negocio defensivo. El portafolio debe tratarlo como posicion ciclica con margen de seguridad, no como refugio.

\section{Financial Profile \& Estimate Quality}

Los datos muestran recuperacion, pero tambien volatilidad. FY23 fue excepcional, FY24 fue debil, FY25 recupero margen y FCF. El consenso FY26E anticipa revenue de USD 7.98bn y EPS de USD 11.20, lo que da soporte a la pantalla de valor. La calidad de estimacion es media porque depende de margenes de refinacion, utilization, capital de trabajo y costos regulatorios.

{table(["Metrica", "FY23", "FY24", "FY25", "FY26E", "Lectura Value"], [
["Revenue (USD m)", "8,232", "7,974", "7,465", "7,975", "Ventas estables, margen manda."],
["Gross margin", "10.1\\%", "1.9\\%", "7.0\\%", "--", "Ciclo visible."],
["Operating margin", "8.9\\%", "0.5\\%", "5.7\\%", "--", "Recuperacion vs FY24."],
["FCF (USD m)", "588.5", "1.7", "356.4", "--", "FCF atractivo pero volatil."],
["ROIC", "39.2\\%", "-1.3\\%", "13.9\\%", "--", "Retorno normalizado incierto."],
["EPS GAAP", "9.65", "-0.59", "7.16", "11.20", "Consenso favorable."],
], "lrrrrX", "scriptsize")}

Q1 2026 confirmo recuperacion operativa con net income atribuible de USD 54.5m, EPS diluido de USD 1.10, adjusted EPS de USD 0.78 y adjusted EBITDA de USD 91.5m. Aun asi, FactSet mostro FCF trimestral negativo por working capital y capex. Ese contraste es central: PARR puede reportar EBITDA positivo y consumir caja temporalmente. Para Value, la tesis solo funciona si esa presion se revierte.

\section{Margin of Safety \& Value Discipline}

La metodologia Value premia descuento, FCF, balance y catalizadores, pero penaliza riesgo de value trap. PARR cumple baratura y capital allocation; falla parcialmente en recurrencia y visibilidad. La accion tiene upside si el mercado aplica un descuento excesivo por ciclicidad, pero el margen de seguridad no es robusto si se usa un EBITDA normalizado conservador.

{score_table([
["Margen de seguridad", "20", "12", "Multiplo bajo, pero upside depende de EBITDA normalizado."],
["Calidad de flujo", "15", "9", "FCF FY25 fuerte; FCF trimestral puede ser erratico."],
["Rentabilidad y ROIC", "10", "6", "ROIC recuperado, pero no persistente."],
["Balance y solvencia", "15", "11", "Liquidez amplia y leverage manejable."],
["Catalizadores", "10", "8", "Recompras, normalizacion de working capital y renewables."],
["Riesgo de value trap", "15", "8", "Principal riesgo: comprar pico de margen."],
["Macro y liquidez", "15", "15", "Entorno de productos refinados puede sostener spreads; liquidez corporativa ayuda."],
])}

\section{Valuation, Reverse DCF \& Credit}

La valuacion de pantalla es atractiva: P/E LTM de 7.19x, EV/EBITDA LTM de 5.33x, EV/Sales LTM de 0.52x y FCF yield FY25 cercano a 11\%. Frente al peer group, PARR cotiza con descuento frente a refinadores de mayor escala. Parte de ese descuento esta justificado por liquidez, tamano, concentracion regional y volatilidad; la oportunidad existe si el descuento excede esos riesgos.

{table(["Variable", "PARR", "Referencia", "Implicacion Value"], [
["P/E LTM", "7.19x", "Bajo absoluto", "Baratura visible."],
["EV/EBITDA LTM", "5.33x", "Bajo vs grandes peers", "Descuento por ciclo y escala."],
["EV/Sales LTM", "0.52x", "Bajo", "Mercado descuenta margen volatil."],
["Target medio", "USD 72", "+11.9\\%", "Upside moderado, no enorme."],
["Net debt/EBITDA FY25", "1.42x", "Conservador", "Balance no invalida tesis."],
["Total debt/EBITDA FY25", "1.71x", "Manejable", "Riesgo de refinanciacion acotado."],
], "lrrX", "scriptsize")}

El reverse DCF cualitativo muestra una zona de decision. Si EBITDA sostenible se mantiene cerca de USD 600m, la accion conserva margen de seguridad razonable. Si EBITDA normalizado cae hacia USD 500m, el multiple ajustado se acerca a una zona menos barata y la tesis depende de recompras y timing. Credito no es el problema principal: deuda total FY25 cercana a USD 803m y liquidez Q1 cercana a USD 938m dan margen operativo.

\section{Risks \& Thesis Breakers}

Los riesgos clave son normalizacion de cracks, working capital negativo, turnarounds, paros no planeados, regulacion ambiental, costos de RINs/CCA, Hawaii concentration, refinanciamiento, importaciones y capital allocation pro-ciclico. La tesis se invalida si EBITDA normalizado cae debajo de USD 500m, si CFO negativo se vuelve recurrente, si recompras se financian con deuda, o si Hawaii Renewables consume capital sin retorno visible.

\section{Portfolio Decision}

La recomendacion es \textbf{Value condicionado / Watchlist activa}. PARR puede incluirse tacticamente si el comite acepta riesgo ciclico y busca exposicion downstream con descuento y catalizador de recompras. El peso debe ser moderado. Una inclusion mas agresiva exige confirmacion de FCF post-turnaround, working capital normalizado y disciplina de capital allocation.
""")


def parr_quality_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Market Cap","USD 3.23bn","FactSet, cierre 8-may-2026"),
("EV/EBITDA LTM","5.33x","Valuacion baja por ciclicidad"),
("Liquidity Q1","USD 938m","Flexibilidad operativa"),
("ROIC FY25","13.9\\%","Recuperacion vs FY24"),
("FCF FY25","USD 356m","Conversion atractiva"),
("Score Quality","58/100","Quality dudosa / Watchlist")
])}

Par Pacific no clasifica como Quality invertible. El negocio tiene activos reales, posicion regional, integracion y liquidez razonable, pero no cumple los filtros principales de Quality: persistencia de margenes, estabilidad de ROIC, recurrencia de flujo y bajo riesgo operacional. La compania puede ser interesante bajo Value o Growth tactico, pero bajo Quality su perfil es demasiado ciclico.

La recomendacion es \textbf{no incluir PARR como Quality core}. Mantenerlo en watchlist permite aprovechar informacion incremental sobre FCF, turnarounds, balance y capital allocation, pero el activo no debe ocupar el espacio de una empresa con moat duradero, alta visibilidad y retornos estables.

\section{Investment View}

PARR es una plataforma downstream integrada con refinerias, retail y logistica. Su atractivo viene de activos fisicos en mercados regionales complejos, no de una ventaja intangible global. Bajo Quality, el comite debe preguntar si la compania puede sostener retornos superiores en multiples ciclos. La evidencia historica no lo confirma: FY23 fue muy fuerte, FY24 fue debil y FY25 recupero parcialmente.

La distincion es importante para portafolio. Un activo puede ser barato y aun asi no ser Quality; puede generar FCF y aun asi no ser estable. PARR pertenece a esa categoria: buen candidato para valor ciclico, candidato debil para calidad estructural.

\section{Business Model \& Cash Engine}

El cash engine depende de refining spreads, disponibilidad operativa, inventarios, costos ambientales, volumen regional y logistica. Retail y Logistics suavizan volatilidad, pero no la eliminan. La base fisica de activos crea barreras de entrada, aunque tambien exige mantenimiento, capex y gestion de riesgo operacional.

{table(["Dimension", "Evidencia", "Lectura Quality", "Riesgo"], [
["Activos fisicos", "4 refinerias / 219 Mbpd", "Barreras regionales reales", "Paros y capex recurrente."],
["Logistica", "13mm bbl almacenamiento", "Soporte al sistema integrado", "No compensa colapso de cracks."],
["Retail", "Marcas Hele, nomnom, 76", "Flujo menos volatil", "Escala limitada."],
["FCF", "USD 356m FY25", "Caja atractiva", "FY24 casi sin FCF."],
])}

\section{Competitive Position \& Addressable Market}

La ventaja competitiva de PARR es local, no global. Hawaii, Pacific Northwest y Rockies tienen fricciones logisticas y activos dificiles de replicar. Eso da posicionamiento, pero no el tipo de moat que permite predecir margenes por diez anos. La competencia incluye grandes refinadores, distribuidores, traders, importaciones y alternativas energeticas.

El mercado direccionable es maduro y regulado. La transicion energetica, obligaciones ambientales y cambios de demanda de combustibles pueden reducir durabilidad. En Quality, un TAM maduro no invalida la tesis si el negocio tiene retornos estables; en PARR, la volatilidad historica impide esa conclusion.

\section{Macro \& Liquidity Backdrop}

PARR puede beneficiarse de tightness de productos refinados, pero Quality no debe depender de tightness. Una desaceleracion ordenada puede mantener demanda razonable, aunque cualquier shock a combustibles, cracks o credito de inventarios impacta caja. La plomeria financiera es manejable por liquidez, pero el capital de trabajo puede absorber caja justo cuando precios suben.

La lectura macro es neutral-negativa para Quality: no hay estres de solvencia evidente, pero tampoco hay estabilidad suficiente para convertir la tesis en defensiva.

\section{Financial Profile \& Estimate Quality}

La serie financiera muestra la razon del descuento Quality. ROIC de 39.2\% en FY23, -1.3\% en FY24 y 13.9\% en FY25 no es persistencia; es ciclo. FCF de USD 588.5m, USD 1.7m y USD 356.4m tampoco es recurrencia. La estimacion FY26E puede ser positiva, pero depende de variables externas.

{table(["Metrica", "FY23", "FY24", "FY25", "FY26E", "Lectura Quality"], [
["Revenue (USD m)", "8,232", "7,974", "7,465", "7,975", "Ventas relativamente maduras."],
["Operating margin", "8.9\\%", "0.5\\%", "5.7\\%", "--", "Volatilidad alta."],
["FCF (USD m)", "588.5", "1.7", "356.4", "--", "Caja no recurrente."],
["ROIC", "39.2\\%", "-1.3\\%", "13.9\\%", "--", "Retornos no persistentes."],
["EPS GAAP", "9.65", "-0.59", "7.16", "11.20", "Earnings ciclicos."],
], "lrrrrX", "scriptsize")}

\section{Quality of Earnings, ROIC \& Capital Discipline}

La calidad de earnings es limitada por sensibilidad a spreads y working capital. El punto favorable es que la administracion ha sido disciplinada con recompras en precios bajos, lo que puede crear valor por accion. Sin embargo, Quality exige mas que buena asignacion en un ciclo favorable; exige que el flujo base sea resistente.

{score_table([
["Rentabilidad superior", "20", "10", "ROIC FY25 positivo, pero no estable."],
["Persistencia y estabilidad", "15", "4", "Margenes altamente ciclicos."],
["Calidad de flujo", "20", "9", "FCF fuerte en FY25, casi nulo en FY24."],
["Balance y solvencia", "15", "10", "Liquidez suficiente; leverage manejable."],
["Moat y posicion competitiva", "10", "7", "Ventaja local por activos y logistica."],
["Capital allocation", "10", "8", "Recompras historicamente atractivas."],
["Valuacion y macro", "10", "10", "Precio bajo ayuda, pero no transforma calidad."],
])}

\section{Valuation, Reverse DCF \& Credit}

La valuacion baja evita una conclusion negativa total, pero no eleva la calidad del negocio. P/E LTM de 7.2x y EV/EBITDA LTM de 5.3x son atractivos, aunque el descuento existe precisamente por volatilidad. Un reverse DCF de Quality requeriria que el EBITDA normalizado fuera estable; PARR aun no demuestra esa estabilidad.

Credito es razonable: liquidez Q1 cercana a USD 938m y deuda manejable. Pero la ausencia de un rating publico robusto y el componente de working capital/inventarios limitan el score. La tesis Quality mejoraria si PARR demuestra varios trimestres de FCF positivo despues de turnarounds y sin depender de cracks extraordinarios.

\section{Risks \& Thesis Breakers}

Los riesgos son ciclos de refinacion, paros operativos, regulacion ambiental, costos de mantenimiento, Hawaii concentration, deterioro de demanda, capital allocation pro-ciclico y presion de working capital. La tesis Quality se invalida si FY26 vuelve a mostrar FCF erratico o si recompras se financian con deuda en lugar de excedente real de caja.

\section{Portfolio Decision}

La recomendacion es \textbf{no incluir PARR bajo Quality}. Puede permanecer como candidato Value/Growth tactico, pero no cumple el estandar de estabilidad, moat y recurrencia que una posicion Quality requiere. El comite debe evitar mezclar bajo multiple con alta calidad: en PARR, el descuento es real, pero la calidad estructural aun no lo es.
""")


def aapl_quality_body():
    return interpolate(r"""
\section{Executive Summary}

{metric_boxes([
("Market Cap","USD 4.31tn","FactSet Snapshot, cierre 8-may-2026"),
("FCF LTM","USD 129.2bn","Motor de capital return"),
("ROIC FY25","70.6\\%","Spread excepcional vs WACC 9.05\\%"),
("Gross Margin FY25","46.9\\%","Mejora estructural por Services"),
("S\\&P Rating","AA+","Credito defensivo"),
("Score Quality","89/100","Quality alta conviccion")
])}

Apple clasifica como \textbf{Quality alta conviccion}. Aunque bajo Value no ofrece margen de seguridad suficiente, bajo Quality presenta uno de los perfiles mas fuertes del universo de mega-cap technology: marca global, ecosistema cerrado, base instalada, pricing power, servicios de alto margen, FCF masivo, ROIC extraordinario, balance resiliente y recompra estructural de acciones.

La recomendacion es inclusion como Quality core, con control de valuacion. El riesgo no es solvencia ni calidad de flujo; el riesgo es pagar demasiado por una franquicia excelente. Aun asi, la durabilidad del moat y la conversion de caja justifican clasificar Apple como activo Quality, no como Value.

\section{Investment View}

Apple fue fundada en 1976 y cotiza en NASDAQ bajo el ticker AAPL. Opera en hardware premium, software, servicios digitales, wearables, pagos, contenido, semiconductores propios y dispositivos personales. Sus marcas y productos principales incluyen iPhone, Mac, iPad, Apple Watch, AirPods, Vision Pro, App Store, iCloud, Apple Music, Apple Pay, AppleCare y Apple TV+.

Bajo Quality, la pregunta central no es si la accion esta barata, sino si la empresa puede sostener retornos superiores durante muchos anos. Apple responde afirmativamente por integracion vertical, lealtad de usuarios, switching costs, developer ecosystem, control de hardware/software, privacidad y escala global. La calidad de la franquicia compensa parte de la madurez del crecimiento.

\section{Business Model \& Cash Engine}

El cash engine de Apple tiene dos capas. La primera es hardware premium, especialmente iPhone, que genera volumen, margen y base instalada. La segunda es Services, que monetiza esa base instalada con ingresos mas recurrentes y margenes superiores. En FY2025, iPhone represento 50.4\% de ingresos y Services 26.2\%, pero Services tuvo margen bruto de 75.4\%, muy superior al de productos.

{table(["Unidad", "\\% ingresos FY25", "Ingresos USD m", "Crec. YoY"], [
["iPhone", "50.4\\%", "209,586", "4.2\\%"],
["Services", "26.2\\%", "109,158", "13.5\\%"],
["Wearables, Home \\& Accessories", "8.6\\%", "35,686", "-3.6\\%"],
["Mac", "8.1\\%", "33,708", "12.4\\%"],
["iPad", "6.7\\%", "28,023", "5.0\\%"],
], "lrrr")}

La calidad del modelo esta en que el hardware crea la base y Services eleva margen, recurrencia y lifetime value. Un usuario de iPhone no solo compra un dispositivo; entra en una red de apps, pagos, almacenamiento, suscripciones, accesorios y datos sincronizados. Esa combinacion sostiene pricing power y reduce churn.

\section{Competitive Position \& Addressable Market}

Apple compite con Samsung, Google, Microsoft, Huawei, Xiaomi, Meta, Amazon, Spotify, Netflix, PayPal y multiples fabricantes de hardware. Su moat no depende de ser el mas barato, sino de experiencia integrada, confianza, privacidad, diseno, chips propios, retail, soporte y ecosistema de desarrolladores.

{table(["Region FY25", "\\% ingresos", "Ingresos USD m", "Crec. YoY"], [
["United States", "36.5\\%", "151,790", "6.7\\%"],
["Europe", "26.7\\%", "111,032", "9.6\\%"],
["Greater China", "15.5\\%", "64,377", "-3.8\\%"],
["Rest of Asia Pacific", "8.1\\%", "33,696", "9.9\\%"],
["Japan", "6.9\\%", "28,703", "14.6\\%"],
["Americas ex-US", "6.4\\%", "26,563", "6.9\\%"],
], "lrrr")}

El TAM es maduro en smartphones, pero amplio en servicios, pagos, salud, wearables, AI on-device, spatial computing y suscripciones. Greater China sigue siendo el principal foco competitivo y geopolitico; aun asi, la diversificacion geografica y la fuerza de marca mitigan la dependencia.

\section{Macro \& Liquidity Backdrop}

En desaceleracion ordenada, Apple funciona como Quality defensivo dentro de tecnologia: tiene FCF actual, balance AA+, baja probabilidad de estres financiero y demanda de reposicion relativamente resiliente. La plomeria financiera favorece a companias con liquidez propia, porque tasas reales altas penalizan activos que dependen de financiamiento externo. Apple no necesita el mercado de capitales para sobrevivir ni para invertir.

El riesgo macro esta en valuacion. Tasas reales cercanas a 2\% reducen tolerancia a multiples altos y hacen que el FCF yield de cerca de 3\% luzca menos generoso. Por eso, la conclusion Quality no equivale a ignorar precio: Apple puede ser excelente empresa y aun asi requerir disciplina de entrada.

\section{Financial Profile \& Estimate Quality}

Apple combina crecimiento moderado con rentabilidad extraordinaria. FY2025 mostro ingresos de USD 416.2bn, EBIT de USD 133.1bn, utilidad neta de USD 112.0bn y FCF de USD 98.8bn. FactSet proyecta ingresos de USD 474.0bn en FY2026, USD 512.5bn en FY2027 y USD 545.0bn en FY2028.

{table(["Metrica", "FY23", "FY24", "FY25", "FY26E", "FY27E", "FY28E"], [
["Ingresos (USD m)", "383,285", "391,035", "416,161", "474,033", "512,521", "544,972"],
["EBIT (USD m)", "114,301", "123,216", "133,050", "153,749", "164,701", "178,670"],
["Net income (USD m)", "96,995", "93,736", "112,010", "127,377", "137,235", "149,203"],
["EPS diluido", "6.13", "6.08", "7.47", "8.69", "9.55", "10.53"],
["FCF (USD m)", "99,584", "108,807", "98,767", "139,674", "148,874", "163,636"],
], "lrrrrrr", "scriptsize")}

La calidad de estimacion es alta por cobertura amplia, disclosure historico, base instalada y visibilidad de segmentos. La debilidad relativa es que el crecimiento no es explosivo; es un compounder de escala, no una historia early-stage.

\section{Quality of Earnings, ROIC \& Capital Discipline}

El bloque Quality es excepcional. Margen bruto subio de 44.1\% en FY2023 a 46.9\% en FY2025, margen operativo se mantuvo cerca de 32\%, FCF margin fue 23.7\% en FY2025 y ROIC alcanzo 70.6\%. La compania convierte utilidad en caja y devuelve capital a accionistas mediante recompras y dividendos.

{table(["Ratio", "FY21", "FY22", "FY23", "FY24", "FY25"], [
["Gross margin", "41.8\\%", "43.3\\%", "44.1\\%", "46.2\\%", "46.9\\%"],
["Operating margin", "29.8\\%", "30.3\\%", "29.8\\%", "31.5\\%", "32.0\\%"],
["Net margin", "25.9\\%", "25.3\\%", "25.3\\%", "24.0\\%", "26.9\\%"],
["FCF margin", "25.4\\%", "28.3\\%", "26.0\\%", "27.8\\%", "23.7\\%"],
["ROIC", "53.4\\%", "58.2\\%", "59.0\\%", "58.2\\%", "70.6\\%"],
], "lrrrrr", "small")}

{score_table([
["Rentabilidad superior", "20", "20", "ROIC de 70.6\\% y margen operativo superior."],
["Persistencia y estabilidad", "15", "14", "Base instalada y Services elevan recurrencia."],
["Calidad de flujo", "20", "19", "FCF masivo y conversion consistente."],
["Balance y solvencia", "15", "15", "AA+, net debt/EBITDA bajo y liquidez amplia."],
["Moat y posicion competitiva", "10", "10", "Marca, ecosistema, chips, privacidad y switching costs."],
["Capital allocation", "10", "8", "Recompras estructurales; penaliza precio alto."],
["Valuacion y macro", "10", "3", "Multiplo premium y FCF yield bajo limitan upside."],
])}

\section{Valuation, Reverse DCF \& Credit}

La valuacion es el unico bloque debil. P/E LTM de 35.5x, EV/EBITDA LTM de 26.9x y FCF yield cercano a 3.0\% son exigentes. Bajo Quality, esto no destruye la tesis, pero si limita peso incremental. El reverse DCF cualitativo exige que Apple sostenga crecimiento de FCF, margen alto, recompras y expansion de Services. Esa exigencia es razonable, pero no deja gran margen si China, AI o regulacion decepcionan.

Credito es una fortaleza clara: S\&P AA+, deuda total de USD 84.7bn, caja e inversiones de corto plazo de USD 68.5bn, net debt/EBITDA de 0.1x y Altman Z-Score de 11.1. Apple no enfrenta riesgo financiero relevante; el debate es retorno esperado desde precio actual.

\section{Risks \& Thesis Breakers}

Riesgos: valuacion premium, regulacion App Store, China, ciclo de iPhone, competencia en AI, presion en servicios, cadena de suministro y recompras a multiples altos. La tesis Quality se invalida si Services pierde margen/crecimiento, si ROIC cae de forma estructural, si China acelera deterioro o si la compania empieza a sacrificar privacidad/experiencia para competir en AI sin monetizacion clara.

\section{Portfolio Decision}

La decision es \textbf{incluir Apple como Quality core con disciplina de precio}. El activo no cumple Value estricto, pero si cumple sobradamente Quality. En portafolio debe funcionar como compounder defensivo de tecnologia, con peso controlado por valuacion y monitoreo de Services, FCF yield, China, margen bruto y recompras.
""")


common_parr = {
    "company": "Par Pacific Holdings, Inc. (PARR)",
    "ticker": "PARR",
    "exchange": "NYSE",
    "sector": "Energy downstream, refining, retail \\& logistics",
}
common_aapl = {
    "company": "Apple Inc. (AAPL)",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "sector": "Consumer technology, hardware, software \\& services",
}

memos = [
    {**common_parr, "philosophy": "Value", "preview": "Value condicionado: descuento visible, pero riesgo de ciclo y value trap.", "file": "PARR_value_investment_memo.tex", "body": parr_value_body()},
    {**common_parr, "philosophy": "Growth", "preview": "Growth Watchlist: recuperacion ciclica con FCF, no compounder secular.", "file": "PARR_growth_investment_memo.tex", "body": parr_growth_body()},
    {**common_parr, "philosophy": "Quality", "preview": "No Quality core: activos buenos, pero flujo y ROIC demasiado ciclicos.", "file": "PARR_quality_investment_memo.tex", "body": parr_quality_body()},
    {**common_aapl, "philosophy": "Quality", "preview": "Quality alta conviccion: ecosistema, ROIC, FCF y balance defensivo.", "file": "AAPL_quality_investment_memo.tex", "body": aapl_quality_body()},
]

for memo in memos:
    render(memo)
