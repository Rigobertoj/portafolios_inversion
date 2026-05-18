from pathlib import Path
import os
import subprocess

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


OUT = Path(__file__).resolve().parent


MAIN_TEX = r"""
\documentclass[12pt,letterpaper]{article}
\usepackage[margin=0.82in]{geometry}
\usepackage{fontspec}
\setmainfont{Garamond}[
  Path=/mnt/c/Windows/Fonts/,
  UprightFont=GARA.TTF,
  BoldFont=GARABD.TTF,
  ItalicFont=GARAIT.TTF
]
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{setspace}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{ragged2e}

\definecolor{navy}{HTML}{173B68}
\definecolor{deepnavy}{HTML}{0B2742}
\definecolor{steel}{HTML}{6E7F92}
\definecolor{softgray}{HTML}{F2F4F7}
\definecolor{linegray}{HTML}{C8D0DA}

\hypersetup{colorlinks=true, linkcolor=navy, urlcolor=navy}
\setstretch{1.25}
\justifying
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\renewcommand{\arraystretch}{1.28}
\setlist[itemize]{leftmargin=1.1em, itemsep=2pt, topsep=2pt}

\titleformat{\section}{\Large\bfseries\color{navy}}{}{0em}{}
\titleformat{\subsection}{\large\bfseries\color{deepnavy}}{}{0em}{}
\titlespacing*{\section}{0pt}{16pt}{8pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

\pagestyle{fancy}
\fancyhf{}
\lhead{\small\color{steel} PARR | Value Investment Memo}
\rhead{\small\color{steel} Mayo 2026}
\cfoot{\small\color{steel} Documento de research - metodologia Value | \thepage}
\renewcommand{\headrulewidth}{0pt}

\newcommand{\contentsrow}[3]{
\noindent\begin{tabularx}{\textwidth}{@{}p{0.04\textwidth}Xr@{}}
{\bfseries\color{navy} #1} & {\bfseries\color{navy} #2} & {\bfseries #3}
\end{tabularx}\vspace{1.05em}
}

\newcommand{\metricbox}[3]{
\begin{tabular}{>{\centering\arraybackslash}p{0.28\textwidth}}
\rowcolor{navy}\color{white}\bfseries #1 \\
\fcolorbox{navy}{white}{\begin{minipage}[c][0.58in][c]{0.255\textwidth}\centering{\Large\bfseries\color{navy} #2}\\[-1pt]{\scriptsize #3}\end{minipage}}
\end{tabular}
}

\begin{document}
\begin{titlepage}
\pagecolor{deepnavy}
\color{white}
\vspace*{0.55in}
{\Huge\bfseries Par Pacific Holdings, Inc. (PARR)}\\[8pt]
{\Large Investment Memo Value}\\[22pt]
{\large Analisis de seleccion de activos bajo filosofia Value}\\[1.15in]
\begin{tabularx}{\textwidth}{X r}
NYSE: PARR & 10 de mayo de 2026 \\
Sector: Energia & Subsector: Refinacion, marketing y logistica \\
Documento principal & Fuentes auditables en anexo separado \\
\end{tabularx}
\vfill
{\large Dictamen preliminar: Value invertible condicionado / Watchlist activa}\\[4pt]
{\small Enfoque: margen de seguridad, FCF normalizado, ciclo de refinacion, balance y riesgo de value trap.}
\end{titlepage}
\nopagecolor
\color{black}

\thispagestyle{fancy}
\section*{Contents}
\vspace{1.5em}
\contentsrow{1}{Executive Summary}{3}
\contentsrow{2}{Investment View}{3}
\contentsrow{3}{Business Model \& Cash Engine}{4}
\contentsrow{4}{Competitive Position \& Addressable Market}{5}
\contentsrow{5}{Macro \& Liquidity Backdrop}{5}
\contentsrow{6}{Financial Profile \& Estimate Quality}{6}
\contentsrow{7}{Quality of Earnings, ROIC \& Capital Return}{6}
\contentsrow{8}{Valuation, Reverse DCF \& Credit}{7}
\contentsrow{9}{Risks \& Thesis Breakers}{8}
\contentsrow{10}{Portfolio Decision}{8}
\newpage

\section{Executive Summary}
\begin{center}
\metricbox{Market Cap}{USD 3.23bn}{FactSet, cierre 8-may-2026}\hfill
\metricbox{EV / EBITDA LTM}{5.33x}{Descuento vs refinadores de gran escala}\hfill
\metricbox{P/E LTM}{7.19x}{Multiplo bajo, pero ciclico}\\[12pt]
\metricbox{FCF FY25}{USD 356m}{11.0\% sobre market cap actual}\hfill
\metricbox{ROIC FY25}{13.9\%}{Recuperacion vs -1.3\% en FY24}\hfill
\metricbox{Liquidez Q1 26}{USD 938m}{Balance flexible para ciclo y recompras}
\end{center}

Par Pacific Holdings presenta una tesis Value mas interesante que la de una compania simplemente barata por multiplo. La accion cotiza a 7.2x P/E LTM y 5.3x EV/EBITDA LTM, con una capitalizacion cercana a USD 3.23bn y Enterprise Value de USD 3.89bn. La lectura inicial favorece una investigacion Value porque el negocio genero USD 356m de FCF en FY25, mantiene liquidez cercana a USD 938m y opera activos fisicos en mercados donde la logistica limita la competencia directa.

El atractivo no proviene de estabilidad defensiva, sino de una posible subvaloracion de activos integrados en nichos regionales. PARR combina refinacion, retail y logistica en Hawaii, Washington, Wyoming y Montana. Esa integracion permite capturar margen a lo largo de la cadena, pero tambien expone la tesis a la volatilidad de cracks, inventarios, RINs, turnarounds, diferenciales de crudo y normalizacion del margen de refinacion.

La recomendacion es \textbf{inclusion condicionada en watchlist Value}, con sesgo positivo si el comite acepta riesgo ciclico. La accion no debe tratarse como compounder defensivo; debe tratarse como activo de ciclo con balance razonablemente sobrevivible, FCF potencialmente alto y catalizadores de capital allocation. La entrada seria mas defendible si el precio ofrece mayor margen de seguridad frente a una normalizacion de EBITDA, o si la compania confirma conversion de caja despues de los turnarounds y la puesta en marcha de Hawaii Renewables.

\section{Investment View}
La tesis central es que PARR puede ser un Value fundamental, pero no un Value automatico. El descuento de multiplos es visible, pero el comite debe separar tres capas: baratura estadistica, calidad economica del activo y capacidad de monetizar el ciclo sin destruir capital. En PARR, la primera capa es clara; la segunda es razonable por integracion logistica y presencia en mercados complejos; la tercera depende de disciplina operativa y de que los margenes actuales no sean transitorios en exceso.

El argumento positivo se sostiene en cuatro puntos. Primero, la compania controla cuatro refinerias con 219 Mbpd de capacidad y una red logistica de 13 millones de barriles de almacenamiento, lo que crea relevancia local y ventajas de abastecimiento en mercados con fricciones. Segundo, el negocio no es refinacion pura: retail y logistica suavizan parcialmente la volatilidad del segmento refining. Tercero, la recompra de acciones ha sido material; desde el inicio del programa, la administracion reporta mas de 14 millones de acciones recompradas, equivalentes a mas de 20\% de las acciones en circulacion. Cuarto, el consenso 2026E implica EPS de USD 11.20 y precio objetivo medio de USD 72, lo que deja upside moderado frente al precio de USD 64.37 usado en el data room.

La objecion principal es que un refinador puede parecer barato justo en el pico del ciclo. Por eso, el memo no concluye ``comprar por P/E bajo''. La pregunta correcta es: que EBITDA normalizado justifica el EV actual despues de ajustar por deuda, capex, mantenimiento, capital de trabajo y riesgo regulatorio. Bajo ese filtro, PARR merece seguimiento activo y podria entrar como posicion Value tactica, pero no como posicion core de baja volatilidad.

Desde el punto de vista corporativo, PARR es una compania publica estadounidense listada en NYSE bajo el ticker PARR. El antecedente corporativo se remonta a una incorporacion en 1984, aunque la plataforma operativa actual se consolida principalmente desde 2012-2013, cuando la compania comenzo a adquirir y desarrollar activos de energia downstream en Hawaii y el oeste de Estados Unidos. Esta historia importa para el comite porque PARR no debe leerse como refinador integrado de gran escala, sino como una plataforma regional construida por adquisiciones, optimizacion operativa y asignacion disciplinada de capital.

\section{Business Model \& Cash Engine}
Par Pacific proporciona combustibles convencionales y renovables al oeste de Estados Unidos. Opera tres segmentos principales: Refining, Retail y Logistics. Refining es el motor de utilidades y volatilidad; Retail aporta cercania al cliente final mediante estaciones y tiendas; Logistics captura rentas de infraestructura, almacenamiento, transporte y distribucion en mercados donde la geografia importa.

El segmento Refining posee refinerias en Kapolei, Hawaii; Newcastle, Wyoming; Tacoma, Washington; y Billings, Montana. Estas instalaciones convierten crudo en gasolina, destilados, asfalto y otros productos. El valor agregado no esta solo en procesar crudo, sino en operar donde la alternativa de suministro es limitada, costosa o logisticamente compleja. En Hawaii, por ejemplo, la isla y la distancia crean una estructura distinta a la de refinadores continentales; en Rockies y Pacific Northwest, la combinacion de activos, ductos, rail y terminales puede sostener capturas superiores cuando el mercado esta ajustado.

Retail opera puntos de venta de combustible y conveniencia en Hawaii, Washington e Idaho bajo marcas como Hele, nomnom y 76. Esta division no transforma por completo la naturaleza ciclica de PARR, pero si mejora la lectura del modelo porque agrega margen de distribucion, contacto con demanda local y monetizacion de marcas regionales.

Logistics es el pegamento economico del sistema. Incluye single point mooring en Hawaii, terminal ferroviaria en Washington, terminales, ductos, camiones, embarcaciones, almacenamiento, racks y activos de movimiento de etanol, crudo y productos refinados. Para una tesis Value, esta division es relevante porque puede sostener EBITDA mas recurrente que Refining y porque reduce la dependencia de terceros en la cadena de suministro.

En escala operativa, el perfil combina aproximadamente 1,758 empleados, 219,000 barriles diarios de capacidad de refinacion y cerca de 13 millones de barriles de almacenamiento. Estos datos dimensionan una compania pequena frente a los lideres de refinacion de Estados Unidos, pero suficientemente relevante en sus mercados regionales. La lectura Value se apoya precisamente en esa asimetria: menor escala bursatil y menor liquidez de mercado pueden generar descuento, mientras los activos fisicos conservan utilidad economica en regiones con restricciones de suministro.

\section{Competitive Position \& Addressable Market}
PARR no compite por escala global contra Valero, Marathon o Phillips 66; compite por posicionamiento local en mercados donde la logistica y la disponibilidad de activos importan mas que el tamano absoluto. Esa diferencia es esencial para no aplicar un peer group mecanico. La compania es pequena frente a los grandes refinadores, pero su presencia en Hawaii, Pacific Northwest y Rockies le permite operar con una logica de nicho.

\begin{center}
\begin{tabularx}{0.98\textwidth}{lrrrX}
\toprule
\textbf{Compania} & \textbf{Market Cap} & \textbf{EV} & \textbf{TEV/EBITDA} & \textbf{Lectura} \\
\midrule
Par Pacific & 3.23bn & 3.08bn & 4.67x & Nicho regional integrado \\
Valero & 71.58bn & 60.17bn & 7.84x & Escala global, mayor liquidez \\
Marathon Petroleum & 71.49bn & 87.22bn & 8.68x & Refinacion + midstream \\
Phillips 66 & 68.78bn & 74.27bn & 10.62x & Diversificacion amplia \\
HF Sinclair & 13.06bn & 10.91bn & 5.75x & Comparable medio mas cercano \\
PBF Energy & 4.82bn & 5.63bn & n.a. & Ciclo y volatilidad elevada \\
\bottomrule
\end{tabularx}
\end{center}

El TAM relevante no debe leerse como mercado global de combustibles, sino como demanda de combustibles, almacenamiento y distribucion en regiones servidas por sus activos. La ventaja competitiva es local, fisica y operacional: ubicacion, permisos, infraestructura, supply chain, conocimiento de mercados insulares o regionales y capacidad de manejar diferenciales de crudo y producto. El moat existe, pero no es intangible ni permanente; puede erosionarse por disrupciones operativas, regulacion ambiental, cambios en demanda de combustibles o inversiones obligatorias de mantenimiento.

\section{Macro \& Liquidity Backdrop}
El entorno macro favorece una lectura selectiva. En una desaceleracion ordenada, los activos ciclicos con balance debil suelen ser penalizados; sin embargo, refinadores con buena liquidez y capacidad de caja pueden capturar episodios de margen elevado cuando la oferta de productos esta ajustada. La EIA, en su STEO de abril de 2026, senala precios de crudo y combustibles mas altos por disrupciones globales, con diesel particularmente ajustado y margenes de refinacion por encima de niveles de 2025.

La plomeria financiera importa porque PARR es sensible al costo de capital, al capital de trabajo y a la disponibilidad de credito basado en inventarios. Un aumento de precios de crudo puede elevar el valor de inventarios y generar presion temporal de working capital, aun cuando el margen economico mejore. El primer trimestre de 2026 lo ilustra: la compania reporto caja operativa afectada por salidas de capital de trabajo, pero tambien indico que mantenia una posicion de liquidez robusta y una expectativa de cash flow favorable.

Para un comite, la conclusion macro no es que ``energia sube''. La conclusion es que PARR puede beneficiarse de tightness en productos refinados, especialmente destilados y mercados Pacific/Rockies, pero el margen de seguridad debe evaluarse contra un escenario de normalizacion de cracks y menor demanda. La accion debe comprarse por flujo normalizado y activos, no por extrapolar un trimestre fuerte.

\section{Financial Profile \& Estimate Quality}
\begin{center}
\begin{tabularx}{0.98\textwidth}{lrrrrX}
\toprule
\textbf{Metrica} & \textbf{FY23} & \textbf{FY24} & \textbf{FY25} & \textbf{FY26E} & \textbf{Lectura} \\
\midrule
Revenue (USD m) & 8,232 & 7,974 & 7,465 & 7,975 & Recuperacion estimada moderada \\
Gross margin & 10.1\% & 1.9\% & 7.0\% & n.a. & Ciclo muy visible \\
Operating margin & 8.9\% & 0.5\% & 5.7\% & n.a. & Mejora fuerte vs FY24 \\
FCF (USD m) & 588.5 & 1.7 & 356.4 & n.a. & Conversion atractiva, no lineal \\
ROIC & 39.2\% & -1.3\% & 13.9\% & n.a. & Retorno normalizado aun incierto \\
EPS GAAP & 9.65 & -0.59 & 7.16 & 11.20 & Consenso exige continuidad \\
\bottomrule
\end{tabularx}
\end{center}

La calidad de las estimaciones es razonable, pero ciclica. El consenso espera que los ingresos FY26 alcancen USD 7.97bn, frente a USD 7.46bn en FY25, y que EPS GAAP llegue a USD 11.20. Esto implica que el mercado no esta pagando un multiple alto por ese crecimiento esperado; el riesgo es que el consenso este capturando margenes de refinacion temporalmente favorables.

El primer trimestre de 2026 fue significativamente mejor que el de 2025: net income atribuible de USD 54.5m, EPS diluido de USD 1.10, adjusted EPS de USD 0.78 y adjusted EBITDA de USD 91.5m. El resultado confirma recuperacion operativa, pero tambien muestra el problema de lectura: el FCF trimestral reportado por FactSet fue negativo por salidas de working capital y capex, mientras la administracion subrayo que parte de esa presion podria revertirse. Para Value, esta diferencia entre utilidad, EBITDA y caja es el punto que debe monitorearse.

\section{Quality of Earnings, ROIC \& Capital Return}
La calidad de las utilidades de PARR es mixta. Por un lado, FY25 muestra FCF de USD 356m, FCF margin de 4.8\%, ROIC de 13.9\% y cobertura EBITDA/intereses de 7.0x. Por otro lado, la serie historica evidencia sensibilidad extrema: en FY24 el ROIC fue negativo, el FCF fue practicamente cero y EV/EBITDA se expandio por caida de EBITDA. La compania tiene capacidad de generar caja, pero esa caja no es equivalente a la de un negocio secular y recurrente.

El capital allocation es uno de los elementos mas favorables de la tesis. La recompra de USD 28m en Q1 2026 a precio promedio de USD 37.96 por accion fue realizada muy por debajo del precio actual de referencia. La administracion tambien reporto que, desde el inicio del programa, ha recomprado mas de 14 millones de acciones a precio promedio cercano a USD 25. Si la compania mantiene disciplina y recompra en periodos de descuento real, la creacion de valor por accion puede ser relevante.

La condicion es que las recompras no compitan contra necesidades de mantenimiento, liquidez y deuda. En refinacion, el capital fisico exige inversiones periodicas. Un programa de buybacks solo agrega valor si se ejecuta despues de preservar balance, seguridad operacional y capacidad de atravesar el ciclo.

\section{Valuation, Reverse DCF \& Credit}
La valuacion luce atractiva en pantalla: 7.2x P/E LTM, 5.3x EV/EBITDA LTM, 0.52x EV/Sales y FCF yield aproximado de 11\% sobre FY25 FCF y market cap actual. Frente al peer group, PARR cotiza con descuento relevante respecto a refinadores de mayor escala. Ese descuento esta parcialmente justificado por menor tamano, menor liquidez, mayor concentracion regional y mayor volatilidad operativa.

\begin{center}
\begin{tabularx}{0.88\textwidth}{lrrX}
\toprule
\textbf{Variable} & \textbf{PARR} & \textbf{Referencia} & \textbf{Implicacion Value} \\
\midrule
P/E LTM & 7.19x & Bajo absoluto & Baratura visible \\
EV/EBITDA LTM & 5.33x & Bajo vs grandes peers & Descuento por ciclo y escala \\
EV/Sales LTM & 0.52x & Bajo & Mercado descuenta margen volatil \\
Target medio & USD 72 & +11.9\% & Upside moderado, no enorme \\
Net debt/EBITDA FY25 & 1.42x & Conservador & Balance no invalida tesis \\
Total debt/EBITDA FY25 & 1.71x & Manejable & Riesgo de refinanciacion acotado \\
\bottomrule
\end{tabularx}
\end{center}

El reverse DCF cualitativo sugiere que el precio actual no requiere una historia de crecimiento secular, pero si requiere que el EBITDA no colapse hacia un escenario de FY24. A USD 3.89bn de EV, un multiple de 5.3x implica EBITDA LTM cercano a USD 730m. Si el EBITDA normalizado fuese mas cercano a USD 500m, el multiple ajustado se aproximaria a 7.8x, menos barato y mas dependiente de catalizadores. Si el EBITDA sostenible se mantiene cerca de USD 600m, la accion conserva margen de seguridad razonable.

En credito, el perfil es aceptable pero no defensivo. La compania no cuenta con rating publico vigente de S\&P o Moody's en el archivo de credito, pero FactSet muestra deuda total de USD 802.9m en FY25 y ratios de leverage manejables. Q1 2026 eleva deuda total a USD 947.6m, mientras la liquidez reportada de USD 937.7m compensa parte del riesgo. La tesis falla si la deuda sube para financiar capital de trabajo permanente o recompras agresivas en precio alto.

\section{Risks \& Thesis Breakers}
\begin{longtable}{p{0.25\textwidth}p{0.43\textwidth}p{0.23\textwidth}}
\toprule
\textbf{Riesgo} & \textbf{Lectura de inversion} & \textbf{Indicador de invalidacion} \\
\midrule
Ciclo de refinacion & Margenes actuales pueden normalizarse si mejora oferta global o cae demanda. & EBITDA normalizado cae bajo USD 500m sin compensacion en caja. \\
Working capital & Precios de crudo y productos pueden absorber caja aun con EBITDA positivo. & CFO negativo recurrente excluyendo solo efectos temporales. \\
Turnarounds y ejecucion & Refinerias requieren mantenimiento; paros no planeados destruyen margen. & Costos de mantenimiento superan guia o afectan throughput varios trimestres. \\
Regulacion ambiental & RINs, CCA, permisos, consent decrees y obligaciones ambientales pueden presionar FCF. & Aumento material de obligaciones sin recuperacion en precio. \\
Hawaii concentration & Mercado atractivo pero expuesto a geografia, logistica y riesgo politico local. & Perdida de ventaja de suministro o intervencion regulatoria adversa. \\
Capital allocation & Buybacks crean valor si se hacen con descuento; destruyen valor si se hacen en pico. & Recompras financiadas con deuda o ejecutadas despues de expansion excesiva de multiples. \\
\bottomrule
\end{longtable}

\section{Portfolio Decision}
El score final ubica a PARR como \textbf{Value invertible condicionado}. La compania cumple con baratura estadistica, presenta una ruta fundamental de valor y no muestra un balance que obligue a excluirla. Sin embargo, no debe aprobarse como posicion core sin monitoreo porque el negocio es ciclico, el FCF trimestral puede ser erratico y la valuacion depende de que el EBITDA normalizado no se comprima de forma severa.

\begin{center}
\begin{tabularx}{0.88\textwidth}{lcX}
\toprule
\textbf{Dimension} & \textbf{Score} & \textbf{Comentario} \\
\midrule
Baratura de multiplos & 8/10 & P/E y EV/EBITDA atractivos vs mercado y peers. \\
Calidad financiera & 6/10 & Buen FY25, pero alta variabilidad ciclica. \\
Balance y liquidez & 7/10 & Liquidez amplia y leverage manejable. \\
Moat / posicion competitiva & 7/10 & Ventaja local por activos y logistica. \\
Riesgo de value trap & 5/10 & Principal riesgo: comprar pico de margen. \\
Capital allocation & 8/10 & Recompras historicamente disciplinadas. \\
\midrule
\textbf{Score ponderado} & \textbf{6.9/10} & \textbf{Watchlist activa con sesgo positivo.} \\
\bottomrule
\end{tabularx}
\end{center}

La recomendacion para portafolio es iniciar cobertura formal y considerar inclusion tactica si el comite busca exposicion Value ciclica a energia downstream con catalizador de recompras. El tamano de posicion debe ser moderado y condicionado a confirmacion de FCF en los proximos trimestres. Una entrada mas agresiva exigiria evidencia de que Hawaii Renewables escala sin consumir capital de manera desproporcionada, que el working capital se normaliza y que la administracion mantiene disciplina de recompra.

\end{document}
"""


SOURCE_TEX = r"""
\documentclass[12pt,letterpaper]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{fontspec}
\setmainfont{Garamond}[
  Path=/mnt/c/Windows/Fonts/,
  UprightFont=GARA.TTF,
  BoldFont=GARABD.TTF,
  ItalicFont=GARAIT.TTF
]
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{setspace}
\usepackage{ragged2e}
\definecolor{navy}{HTML}{173B68}
\setstretch{1.18}
\justifying
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\hypersetup{colorlinks=true,urlcolor=navy,linkcolor=navy}
\begin{document}
{\Huge\bfseries\color{navy} PARR Value Memo}\\[4pt]
{\Large Source Notes \& Audit Trail}\\[10pt]
{\small Documento separado de soporte para auditar las cifras, afirmaciones operativas y fuentes externas utilizadas en el memo principal.}

\section*{Jerarquia De Fuentes}
\begin{longtable}{p{0.18\textwidth}p{0.33\textwidth}p{0.42\textwidth}}
\toprule
\textbf{Nivel} & \textbf{Fuente} & \textbf{Uso en el memo} \\
\midrule
1 & PARR.zip / 10-Q Q1 2026 & Modelo de negocio, segmentos, ubicacion de refinerias, marcas, resultados trimestrales y riesgos regulatorios. \\
1 & PARR.zip / FactSet Fundamentals & Estados financieros, ratios historicos, acciones, market cap, EV, multiplos, FCF y leverage. \\
1 & PARR.zip / Transcript Q1 2026 & Comentarios de administracion sobre throughput, capital allocation, liquidez, Hawaii Renewables y condiciones de margen. \\
2 & Par Pacific Q1 2026 press release & Validacion externa publica de resultados, balance, liquidez, throughput, capacidad y activos logisticos. \\
2 & Metodologia Value proporcionada & Criterios para clasificar Value estadistico, fundamental e invertible; overlay macro-liquidez y auditoria de value trap. \\
3 & EIA STEO April 2026 & Contexto macro de energia: crudo, gasolina, diesel, inventarios y margenes de refinacion. \\
3 & Company profile / StockAnalysis & Fundacion/incorporacion, empleados, ticker, sector, industria y datos corporativos de referencia. \\
\bottomrule
\end{longtable}

\section*{Notas Auditables Por Dato}
\begin{longtable}{p{0.25\textwidth}p{0.31\textwidth}p{0.36\textwidth}}
\toprule
\textbf{Dato usado} & \textbf{Fuente precisa} & \textbf{Observacion de uso} \\
\midrule
Market Cap USD 3.23bn & PARR/Overview/Companyscreenings\_Results\_report\_.xlsx, sheet PARR-US, row 24; cierre 8-may-2026 row 42. & Usado en KPI inicial y FCF yield aproximado. \\
Enterprise Value USD 3.89bn & Mismo archivo, sheet PARR-US, row 26. & Usado en valuacion y lectura de EV/EBITDA. \\
P/E LTM 7.19x & Mismo archivo, row 27. & Usado como indicador de baratura estadistica. \\
EV/Sales LTM 0.52x y EV/EBITDA LTM 5.33x & Mismo archivo, rows 29-30. & Usado para comparacion Value. \\
Target price medio USD 72 y rating Overweight & PARR/Estimation/target\_rating\_20260510\_20FQ.xlsx, sheet PARR-US, row 4. & Usado para upside moderado de consenso. \\
EPS FY26E USD 11.20 & PARR/Estimation/estimation\_summary\_20260510\_20FQ.xlsx, sheet PARR-US, row 25. & Usado en lectura de expectativas. \\
Revenue FY25 USD 7.465bn y FY26E USD 7.975bn & Mismo archivo, rows 62-63. & Usado en tabla financiera. \\
FY25 FCF USD 356.4m & PARR/Financial/ratio\_analysis\_20260510\_20FQ.xlsx, sheet PARR-US, row 16 con revenue FY25; tambien Credit Overview row 31. & Base de FCF yield aproximado y calidad de caja. \\
ROIC FY25 13.9\%, ROIC FY24 -1.3\% & PARR/Financial/ratio\_analysis\_20260510\_20FQ.xlsx, row 23. & Usado en calidad de utilidades. \\
Net debt/EBITDA FY25 1.42x; total debt/EBITDA FY25 1.71x & PARR/Financial/ratio\_analysis\_20260510\_20FQ.xlsx, rows 90 y 92. & Usado en credito y balance. \\
EBITDA/Interest Expense FY25 7.0x & PARR/Financial/ratio\_analysis\_20260510\_20FQ.xlsx, row 94. & Usado en salud financiera. \\
Acciones en circulacion 49.27m Q1 2026 & PARR/Financial/reported\_shares\_20260510\_20FQ.xlsx, row 10. & Usado para escala bursatil y recompras. \\
Q1 2026 net income USD 54.5m; EPS diluido USD 1.10 & PARR/Earnings/PAR PACIFIC HOLDINGS, INC. files (10-Q) Basic quarterly filin.pdf, statement of operations; tambien press release Q1 2026. & Usado en resultados recientes. \\
Tres segmentos: Refining, Retail, Logistics & 10-Q Q1 2026, page 8, Note 1 Overview. & Base de modelo de negocio. \\
Cuatro refinerias: Kapolei, Newcastle, Tacoma, Billings & 10-Q Q1 2026, page 8, Note 1 Overview. & Base de geografia y activos. \\
Marcas Hele, nomnom y 76 & 10-Q Q1 2026, page 8, Note 1 Overview. & Base de marcas asociadas. \\
Red logistica multimodal & 10-Q Q1 2026, page 8, Note 1 Overview. & Base de moat logistico. \\
Recompra Q1 2026 USD 28m a USD 37.96; mas de 14m acciones desde inicio & Corrected Transcript Q1 2026, page 3; Par Pacific Q1 2026 press release. & Usado en capital allocation. \\
Liquidez USD 937.7m y caja USD 172.2m al 31-mar-2026 & Par Pacific Q1 2026 press release; tambien transcript page 3. & Usado en credito y resiliencia. \\
Capacidad 219,000 bpd y 13m barriles de almacenamiento & Par Pacific Q1 2026 press release, About Par Pacific. & Usado en escala operativa. \\
Throughput Q1 2026 184.3 Mbpd y record en Hawaii 89.8 Mbpd & Par Pacific Q1 2026 press release, Operating Statistics; transcript pages 3-4. & Usado en lectura operacional. \\
Hawaii Renewables inicio operaciones comerciales en abril 2026 & Par Pacific Q1 2026 press release; transcript page 4. & Usado como catalizador y riesgo de ejecucion. \\
Fundacion/incorporacion 1984; NYSE PARR; empleados 1,758 & StockAnalysis company profile; FinanceCharts para IPO 5-sep-2012; Par Pacific NYSE listing press release para migracion a NYSE 20-feb-2018. & Usado en contexto corporativo; IPO/NYSE tratado con cautela por historia corporativa. \\
Marco Value: no basta multiplo bajo; exige FCF, balance, moat, catalizador y evitar value trap & 01.Metodologia Value.pdf, pages 1-3. & Base metodologica del dictamen. \\
Macro energia 2026: crudo y diesel altos, inventarios de destilados ajustados, margenes diesel arriba de 2025 & EIA Short-Term Energy Outlook, April 2026, Petroleum products section. & Usado en Macro \& Liquidity Backdrop. \\
\bottomrule
\end{longtable}

\section*{Fuentes Web Consultadas}
\begin{itemize}
\item Par Pacific Q1 2026 Results: \url{https://www.parpacific.com/press-releases/par-pacific-holdings-reports-first-quarter-2026-results}
\item Par Pacific About Us: \url{https://www.parpacific.com/about-us}
\item Par Pacific Our Story: \url{https://www.parpacific.com/about-us/our-story}
\item Par Pacific NYSE listing announcement: \url{https://www.parpacific.com/press-releases/par-pacific-holdings-announces-trading-new-york-stock-exchange}
\item StockAnalysis PARR profile: \url{https://stockanalysis.com/stocks/parr/company/}
\item FinanceCharts IPO date reference: \url{https://www.financecharts.com/stocks/PARR/summary/ipo-date}
\item EIA STEO Petroleum Products: \url{https://www.eia.gov/outlooks/steo/report/petro_prod.php}
\end{itemize}

\end{document}
"""


DOC_SECTIONS = [
    ("Executive Summary", [
        "Par Pacific Holdings presenta una tesis Value mas interesante que la de una compania simplemente barata por multiplo. La accion cotiza a 7.2x P/E LTM y 5.3x EV/EBITDA LTM, con una capitalizacion cercana a USD 3.23bn y Enterprise Value de USD 3.89bn.",
        "La recomendacion es inclusion condicionada en watchlist Value. La accion no debe tratarse como compounder defensivo; debe tratarse como activo de ciclo con balance razonablemente sobrevivible, FCF potencialmente alto y catalizadores de capital allocation.",
    ]),
    ("Investment View", [
        "La tesis central es que PARR puede ser un Value fundamental, pero no un Value automatico. El descuento de multiplos es visible, pero el comite debe separar baratura estadistica, calidad economica del activo y capacidad de monetizar el ciclo sin destruir capital.",
    ]),
    ("Business Model & Cash Engine", [
        "Par Pacific opera Refining, Retail y Logistics. Refining es el motor de utilidades y volatilidad; Retail aporta cercania al cliente final; Logistics captura rentas de infraestructura, almacenamiento, transporte y distribucion.",
    ]),
    ("Competitive Position & Addressable Market", [
        "PARR compite por posicionamiento local en mercados donde la logistica y disponibilidad de activos importan mas que el tamano absoluto. Su moat es fisico, regional y operacional.",
    ]),
    ("Macro & Liquidity Backdrop", [
        "El entorno macro favorece una lectura selectiva. La EIA senala precios de crudo y combustibles mas altos por disrupciones globales, con diesel ajustado y margenes de refinacion por encima de 2025.",
    ]),
    ("Financial Profile & Estimate Quality", [
        "El consenso espera ingresos FY26 de USD 7.97bn y EPS GAAP de USD 11.20. El riesgo es que el consenso capture margenes de refinacion temporalmente favorables.",
    ]),
    ("Quality of Earnings, ROIC & Capital Return", [
        "FY25 muestra FCF de USD 356m, FCF margin de 4.8%, ROIC de 13.9% y cobertura EBITDA/intereses de 7.0x. La variabilidad historica obliga a monitorear conversion de caja.",
    ]),
    ("Valuation, Reverse DCF & Credit", [
        "La valuacion luce atractiva en pantalla: 7.2x P/E LTM, 5.3x EV/EBITDA LTM y FCF yield aproximado de 11% sobre FY25 FCF y market cap actual. El descuento esta parcialmente justificado por menor escala y volatilidad.",
    ]),
    ("Risks & Thesis Breakers", [
        "Los principales riesgos son normalizacion de cracks, working capital, turnarounds, regulacion ambiental, concentracion en Hawaii y recompras ejecutadas sin margen de seguridad.",
    ]),
    ("Portfolio Decision", [
        "PARR clasifica como Value invertible condicionado, con score ponderado de 6.9/10. La inclusion en portafolio debe ser tactica y moderada, condicionada a FCF y disciplina de balance.",
    ]),
]


def write_docx(path: Path, title: str, sections):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.85)
    sec.right_margin = Inches(0.85)
    styles = doc.styles
    styles["Normal"].font.name = "Garamond"
    styles["Normal"].font.size = Pt(12)
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = title_p.add_run(title)
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(23, 59, 104)
    sub = doc.add_paragraph("Investment Memo Value | 10 de mayo de 2026")
    sub.runs[0].font.size = Pt(11)
    sub.runs[0].font.color.rgb = RGBColor(90, 100, 112)
    for heading, paras in sections:
        h = doc.add_heading(heading, level=1)
        h.runs[0].font.color.rgb = RGBColor(23, 59, 104)
        for para in paras:
            p = doc.add_paragraph(para)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    doc.save(path)


def main():
    main_tex = OUT / "PARR_value_investment_memo.tex"
    source_tex = OUT / "PARR_value_source_notes.tex"
    main_tex.write_text(MAIN_TEX.strip() + "\n", encoding="utf-8")
    source_tex.write_text(SOURCE_TEX.strip() + "\n", encoding="utf-8")
    write_docx(OUT / "PARR_value_investment_memo.docx", "Par Pacific Holdings, Inc. (PARR)", DOC_SECTIONS)
    write_docx(OUT / "PARR_value_source_notes.docx", "PARR Value Memo - Source Notes", [
        ("Audit Trail", [
            "El documento principal usa el 10-Q Q1 2026, FactSet Fundamentals, transcript Q1 2026, press release Q1 2026, metodologia Value, EIA STEO y perfiles corporativos externos para validar historia, empleados, listado y entorno macro.",
            "La version PDF de source notes contiene el detalle fila por fila y pagina por pagina de las cifras utilizadas.",
        ])
    ])
    env = os.environ.copy()
    env.update({
        "TEXMFVAR": "/tmp/texmf-var",
        "TEXMFCONFIG": "/tmp/texmf-config",
    })
    for tex in [main_tex, source_tex]:
        for _ in range(2):
            subprocess.run(
                ["lualatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=OUT,
                check=True,
                env=env,
            )


if __name__ == "__main__":
    main()
