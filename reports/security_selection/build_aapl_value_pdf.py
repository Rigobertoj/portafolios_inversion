from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
import reportlab.pdfbase.pdfdoc as pdfdoc
from hashlib import md5 as _hashlib_md5


def _md5_compat(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)
    return _hashlib_md5(*args, **kwargs)


pdfdoc.md5 = _md5_compat


OUT = "AAPL_value_investment_memo.pdf"
PAGE_W, PAGE_H = letter

GS_NAVY = colors.HexColor("#1E3D70")
GS_BLUE = colors.HexColor("#7E93B0")
GS_PALE = colors.HexColor("#D8DEE7")
GS_DARK = colors.HexColor("#0A3850")
SOFT = colors.HexColor("#F2F3F5")
TEXT = colors.HexColor("#202428")

pdfmetrics.registerFont(TTFont("Garamond", "/mnt/c/Windows/Fonts/GARA.TTF"))
pdfmetrics.registerFont(TTFont("Garamond-Bold", "/mnt/c/Windows/Fonts/GARABD.TTF"))
pdfmetrics.registerFont(TTFont("Garamond-Italic", "/mnt/c/Windows/Fonts/GARAIT.TTF"))


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "BodyG",
        fontName="Garamond",
        fontSize=12,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=TEXT,
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        "H1G",
        fontName="Garamond-Bold",
        fontSize=18,
        leading=21,
        textColor=GS_NAVY,
        spaceBefore=12,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        "H2G",
        fontName="Garamond-Bold",
        fontSize=14,
        leading=17,
        textColor=GS_NAVY,
        spaceBefore=8,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        "SmallG",
        fontName="Garamond",
        fontSize=9.2,
        leading=11,
        textColor=TEXT,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        "SmallCenter",
        fontName="Garamond",
        fontSize=9.2,
        leading=11,
        alignment=TA_CENTER,
        textColor=TEXT,
    )
)
styles.add(
    ParagraphStyle(
        "KpiTitle",
        fontName="Garamond-Bold",
        fontSize=12,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.white,
    )
)
styles.add(
    ParagraphStyle(
        "KpiValue",
        fontName="Garamond-Bold",
        fontSize=17,
        leading=19,
        alignment=TA_CENTER,
        textColor=GS_NAVY,
    )
)
styles.add(
    ParagraphStyle(
        "KpiSub",
        fontName="Garamond",
        fontSize=8.5,
        leading=10,
        alignment=TA_CENTER,
        textColor=TEXT,
    )
)
styles.add(
    ParagraphStyle(
        "TableCell",
        fontName="Garamond",
        fontSize=8.5,
        leading=10,
        alignment=TA_LEFT,
        textColor=TEXT,
    )
)
styles.add(
    ParagraphStyle(
        "TableHead",
        fontName="Garamond-Bold",
        fontSize=8.7,
        leading=10,
        alignment=TA_LEFT,
        textColor=colors.white,
    )
)


def p(text, style="BodyG"):
    return Paragraph(text, styles[style])


def table(data, widths=None, font_size=9.5):
    wrapped = []
    for r, row in enumerate(data):
        out = []
        for cell in row:
            if isinstance(cell, Paragraph):
                out.append(cell)
            else:
                style = styles["TableHead"] if r == 0 else styles["TableCell"]
                out.append(Paragraph(str(cell), style))
        wrapped.append(out)
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Garamond", font_size),
                ("FONT", (0, 0), (-1, 0), "Garamond-Bold", font_size),
                ("BACKGROUND", (0, 0), (-1, 0), GS_NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9AA6B2")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    t.spaceBefore = 12
    t.spaceAfter = 16
    return t


def kpi(title, value, sub):
    return Table(
        [[p(title, "KpiTitle")], [p(value, "KpiValue")], [p(sub, "KpiSub")]],
        colWidths=[1.9 * inch],
        rowHeights=[0.34 * inch, 0.38 * inch, 0.32 * inch],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GS_NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 1.1, GS_NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        ),
    )


def normal_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Garamond", 9)
    canvas.setFillColor(GS_BLUE)
    canvas.drawString(0.82 * inch, PAGE_H - 0.42 * inch, "AAPL | Value Investment Memo")
    canvas.drawRightString(PAGE_W - 0.82 * inch, PAGE_H - 0.42 * inch, "Mayo 2026")
    canvas.setStrokeColor(GS_PALE)
    canvas.line(0.82 * inch, PAGE_H - 0.52 * inch, PAGE_W - 0.82 * inch, PAGE_H - 0.52 * inch)
    canvas.drawCentredString(PAGE_W / 2, 0.42 * inch, f"Documento de research - metodologia Value | {doc.page}")
    canvas.restoreState()


def cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(GS_DARK)
    canvas.rect(0, 2.15 * inch, PAGE_W, 4.2 * inch, fill=1, stroke=0)
    canvas.setFillColor(GS_PALE)
    canvas.rect(0, 6.35 * inch, PAGE_W, 0.7 * inch, fill=1, stroke=0)
    canvas.setFillColor(GS_BLUE)
    canvas.rect(0, 0, 0.62 * inch, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(GS_NAVY)
    canvas.rect(PAGE_W - 1.25 * inch, 6.35 * inch, 0.55 * inch, 0.7 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Garamond-Bold", 33)
    canvas.drawString(1.0 * inch, 5.45 * inch, "Apple Inc. (AAPL)")
    canvas.setFont("Garamond-Bold", 20)
    canvas.drawString(1.0 * inch, 5.05 * inch, "Value Investment Memo")
    canvas.setFont("Garamond", 14)
    canvas.drawString(1.0 * inch, 4.55 * inch, "Security selection | Hardware & consumer technology")
    canvas.setFont("Garamond", 12)
    canvas.drawRightString(PAGE_W - 0.9 * inch, 5.45 * inch, "10 de mayo de 2026")
    canvas.drawRightString(PAGE_W - 0.9 * inch, 5.15 * inch, "NASDAQ: AAPL")
    canvas.drawRightString(PAGE_W - 0.9 * inch, 4.85 * inch, "Filosofia: Value")
    canvas.setFillColor(GS_NAVY)
    canvas.setFont("Garamond-Bold", 13)
    canvas.drawString(1.0 * inch, 1.45 * inch, "Decision preview")
    canvas.setFont("Garamond", 12)
    canvas.drawString(1.0 * inch, 1.18 * inch, "Watchlist: calidad excepcional, pero sin margen de seguridad suficiente.")
    canvas.setFont("Garamond-Italic", 11)
    canvas.setFillColor(GS_BLUE)
    canvas.drawString(1.0 * inch, 0.72 * inch, "Prepared for portfolio construction and asset selection review.")
    canvas.restoreState()


doc = BaseDocTemplate(
    OUT,
    pagesize=letter,
    leftMargin=0.82 * inch,
    rightMargin=0.82 * inch,
    topMargin=0.72 * inch,
    bottomMargin=0.72 * inch,
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 0.15 * inch, id="normal")
doc.addPageTemplates(
    [
        PageTemplate(id="cover", frames=frame, onPage=cover),
        PageTemplate(id="normal", frames=frame, onPage=normal_page),
    ]
)

story = [NextPageTemplate("normal"), PageBreak()]

story.append(p("Contents", "H1G"))
contents = [
    ("1", "Executive Summary", "3"),
    ("2", "Investment View", "3"),
    ("3", "Business Model & Cash Engine", "4"),
    ("4", "Competitive Position & Addressable Market", "4"),
    ("5", "Macro & Liquidity Backdrop", "5"),
    ("6", "Financial Profile & Estimate Quality", "5"),
    ("7", "Quality of Earnings, ROIC & Capital Return", "6"),
    ("8", "Valuation, Reverse DCF & Credit", "6"),
    ("9", "Risks & Thesis Breakers", "7"),
    ("10", "Portfolio Decision", "7"),
]
toc_rows = []
for num, title, page in contents:
    dots = "." * max(8, 70 - len(title))
    toc_rows.append([Paragraph(f"<b>{num}</b>", styles["BodyG"]), Paragraph(f"<b>{title}</b> {dots}", styles["BodyG"]), Paragraph(f"<b>{page}</b>", styles["BodyG"])])
toc = Table(toc_rows, colWidths=[0.4 * inch, 5.2 * inch, 0.4 * inch])
toc.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Garamond", 12),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ALIGN", (2, 0), (2, -1), "RIGHT"),
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(Spacer(1, 12))
story.append(toc)
story.append(PageBreak())

story.append(p("Executive Summary", "H1G"))
story.append(
    Table(
        [
            [kpi("Market Cap", "USD 4.31tn", "FactSet, cierre 8-may-2026"), kpi("P/E LTM", "35.5x", "Prima elevada para Value"), kpi("FCF LTM", "USD 129.2bn", "FCF yield aprox. 3.0%")],
            [kpi("EV/EBITDA LTM", "26.9x", "Multiplo premium"), kpi("ROIC FY25", "70.6%", "Spread alto vs WACC"), kpi("S&P Rating", "AA+", "Credito defensivo")],
        ],
        colWidths=[2.08 * inch, 2.08 * inch, 2.08 * inch],
        style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]),
    )
)
story.append(Spacer(1, 10))
story.append(p("Apple no clasifica hoy como un activo Value invertible en sentido estricto. La compania presenta una calidad financiera extraordinaria: flujo libre recurrente, retorno sobre capital invertido superior al costo de capital, balance resiliente, marca global, ecosistema cerrado y capacidad de recompras a escala. Sin embargo, el precio actual no ofrece el margen de seguridad minimo que exige una filosofia Value disciplinada. A USD 293.32 por accion, el consenso FactSet implica solo 4.9% de retorno al precio objetivo medio, mientras que los multiplos de mercado se mantienen en zona premium: P/E LTM de 35.5x, EV/Sales LTM de 9.5x y EV/EBITDA LTM de 26.9x."))
story.append(p("El dictamen es <b>Value condicionado / Watchlist</b>, con score de <b>62/100</b>. La pregunta clave para el comite no es si Apple es una gran compania, sino si el precio actual remunera el riesgo de pagar por esa calidad. La respuesta actual es no: el activo puede ser defendible en una cartera por liquidez, baja probabilidad de deterioro y resiliencia de FCF, pero no cumple el umbral de entrada para una tesis Value pura."))
story.append(p("<b>Anticipacion de objecion:</b> si el comite pregunta por que no comprar una empresa con AA+, USD 129bn de FCF LTM y ROIC de 70.6%, la respuesta es que Value separa calidad de precio. AAPL puede ser una posicion de calidad, pero a 35.5x P/E y con upside de consenso de 4.9%, el margen de seguridad es insuficiente."))

sections = [
    ("Investment View", [
        "Apple Inc. fue fundada en 1976 y cotiza en NASDAQ bajo el ticker AAPL. Opera en tecnologia de consumo, servicios digitales, software, semiconductores propios, wearables, pagos y dispositivos personales. Su capitalizacion de mercado era de aproximadamente USD 4.31tn, con enterprise value de USD 4.30tn y beta ajustada a tres anos de 1.07.",
        "La accion se encontraba cerca de su maximo de 52 semanas: FactSet reportaba un rango de USD 193.46 a USD 294.76 y un precio equivalente al 99.5% del maximo. Esta informacion es central para el lente Value: aunque el negocio tenga alta calidad, el punto de entrada no refleja castigo de mercado.",
        "Apple reporto en su comunicado oficial de Q2 fiscal 2026 ingresos trimestrales de USD 111.2bn, crecimiento de 17% anual, EPS diluido de USD 2.01 y una nueva autorizacion de recompra por hasta USD 100bn. Para el comite, el punto relevante es que el momentum fundamental no esta roto; lo que esta en duda es el precio de entrada."
    ]),
    ("Business Model & Cash Engine", [
        "Apple monetiza un ecosistema integrado de hardware, software y servicios. Sus productos principales son iPhone, Mac, iPad, Apple Watch, AirPods, Apple Vision Pro y accesorios; sus servicios incluyen App Store, Apple Music, Apple Pay, iCloud, Apple TV+, AppleCare, publicidad y otros servicios digitales.",
        "El valor agregado proviene de integracion vertical, diseno, privacidad, distribucion global, base instalada, switching costs y monetizacion recurrente sobre dispositivos ya vendidos. El modelo tiene dos capas: hardware premium, especialmente iPhone, y Services, que mejora recurrencia, margen bruto y estabilidad. La implicacion para seleccion es clara: Apple tiene un motor de flujo superior, pero ese motor ya esta capitalizado en el multiple."
    ]),
]

for title, paras in sections:
    story.append(p(title, "H1G"))
    for para in paras:
        story.append(p(para))

story.append(table([
    ["Unidad", "% ingresos FY25", "Ingresos USD m", "Crec. YoY"],
    ["iPhone", "50.4%", "209,586", "4.2%"],
    ["Services", "26.2%", "109,158", "13.5%"],
    ["Wearables, Home & Accessories", "8.6%", "35,686", "-3.6%"],
    ["Mac", "8.1%", "33,708", "12.4%"],
    ["iPad", "6.7%", "28,023", "5.0%"],
], [2.4 * inch, 1.25 * inch, 1.45 * inch, 1.1 * inch]))

more_sections = [
    ("Competitive Position & Addressable Market", [
        "Apple compite en smartphones, computadoras personales, tablets, wearables, contenido, pagos, cloud consumer, publicidad digital, aplicaciones y servicios de suscripcion. Sus competidores directos e indirectos incluyen Samsung, Xiaomi, Huawei, Google, Microsoft, Meta, Amazon, Spotify, Netflix, PayPal y multiples fabricantes de hardware de bajo costo.",
        "El TAM sigue siendo amplio, pero maduro en la capa hardware: reemplazo de smartphones, premiumization, wearables, servicios, pagos, salud, inteligencia artificial on-device y ecosistemas de suscripcion. El moat es profundo por marca, base instalada, integracion, chips propios, privacidad, retail y developer ecosystem. Bajo Value, este moat justifica una prima; no justifica comprar sin margen de seguridad. Si el comite exige un catalizador, AI on-device y Services son los candidatos, pero aun no bastan para redefinir la entrada como Value."
    ])
]
for title, paras in more_sections:
    story.append(p(title, "H1G"))
    for para in paras:
        story.append(p(para))

story.append(table([
    ["Region", "% ingresos FY25", "Ingresos USD m", "Crec. YoY"],
    ["United States", "36.5%", "151,790", "6.7%"],
    ["Europe", "26.7%", "111,032", "9.6%"],
    ["Greater China", "15.5%", "64,377", "-3.8%"],
    ["Rest of Asia Pacific", "8.1%", "33,696", "9.9%"],
    ["Japan", "6.9%", "28,703", "14.6%"],
    ["Americas ex-US", "6.4%", "26,563", "6.9%"],
], [2.2 * inch, 1.35 * inch, 1.45 * inch, 1.2 * inch]))

story.append(p("Macro & Liquidity Backdrop", "H1G"))
story.append(p("El entorno macro al 10 de mayo de 2026 puede describirse como desaceleracion ordenada con liquidez todavia funcional, inflacion no completamente resuelta y tasas reales aun restrictivas para activos de duracion. La BLS reporto para marzo de 2026 un CPI anual de 3.3% y core CPI anual de 2.6%; la publicacion de abril estaba programada para el 12 de mayo. En FRED, la tasa real a 10 anos se ubicaba en 1.94% el 6 de mayo y el breakeven a 10 anos en 2.42%. La NFCI marcaba -0.51 al 1 de mayo, condiciones financieras mas laxas que el promedio."))
story.append(p("Para Apple, el entorno favorece empresas con FCF actual, balance fuerte y pricing power; Apple cumple esos requisitos. Pero tasas reales cercanas a 2% reducen la tolerancia a pagar 30x-35x utilidades por crecimiento de un digito alto. En Value, esta lectura exige mayor margen de seguridad, no menor."))

story.append(p("Financial Profile & Estimate Quality", "H1G"))
story.append(p("Apple combina crecimiento moderado con rentabilidad extrema. En FY2025, los ingresos crecieron 6.4% a USD 416.2bn, el EBIT fue USD 133.1bn y el FCF fue USD 98.8bn. El consenso FactSet proyecta ingresos de USD 474.0bn en FY2026, USD 512.5bn en FY2027 y USD 545.0bn en FY2028. La calidad de la estimacion es razonable por la amplitud de cobertura: 57 brokers contribuyen al rating y target price."))
story.append(table([
    ["Metrica", "FY23", "FY24", "FY25", "FY26E", "FY27E", "FY28E"],
    ["Ingresos (USD m)", "383,285", "391,035", "416,161", "474,033", "512,521", "544,972"],
    ["EBIT (USD m)", "114,301", "123,216", "133,050", "153,749", "164,701", "178,670"],
    ["Net income (USD m)", "96,995", "93,736", "112,010", "127,377", "137,235", "149,203"],
    ["EPS diluido", "6.13", "6.08", "7.47", "8.69", "9.55", "10.53"],
    ["FCF (USD m)", "99,584", "108,807", "98,767", "139,674", "148,874", "163,636"],
], [1.55 * inch, 0.76 * inch, 0.76 * inch, 0.76 * inch, 0.84 * inch, 0.84 * inch, 0.84 * inch], 8.6))

story.append(p("Quality of Earnings, ROIC & Capital Return", "H1G"))
story.append(p("El crecimiento de Apple es de alta calidad, aunque no explosivo. La recurrencia viene de Services, instalada sobre una base masiva de dispositivos activos. El margen bruto total subio de 44.1% en FY2023 a 46.9% en FY2025; el margen operativo se mantuvo cerca de 32%; y el margen FCF de FY2025 fue 23.7%."))
story.append(p("El ROIC de FY2025 fue 70.6%, muy superior al WACC reportado de 9.05%. Este spread economico es la mayor defensa fundamental de Apple. La recompra de acciones tambien ha reducido el conteo accionario: las acciones en circulacion pasaron de 15.12bn en septiembre de 2024 a 14.67bn en marzo de 2026."))
story.append(table([
    ["Ratio", "FY21", "FY22", "FY23", "FY24", "FY25"],
    ["Gross margin", "41.8%", "43.3%", "44.1%", "46.2%", "46.9%"],
    ["Operating margin", "29.8%", "30.3%", "29.8%", "31.5%", "32.0%"],
    ["Net margin", "25.9%", "25.3%", "25.3%", "24.0%", "26.9%"],
    ["FCF margin", "25.4%", "28.3%", "26.0%", "27.8%", "23.7%"],
    ["ROIC", "53.4%", "58.2%", "59.0%", "58.2%", "70.6%"],
], [1.7 * inch, 0.88 * inch, 0.88 * inch, 0.88 * inch, 0.88 * inch, 0.88 * inch]))

story.append(p("Valuation, Reverse DCF & Credit", "H1G"))
story.append(p("El bloque de valuacion es el punto debil del caso Value. Apple cotiza a P/E LTM de 35.5x, P/S de 9.6x, EV/Sales de 9.5x y EV/EBITDA de 26.9x. En terminos de FCF, el LTM FCF de USD 129.2bn frente a market cap de USD 4.31tn implica FCF yield aproximado de 3.0%. Este rendimiento de caja no compensa de forma evidente frente a una tasa real de 10 anos cercana a 1.94% y un nominal implicito aproximado de 4.36% si se suma breakeven de 2.42%."))
story.append(p("El reverse DCF cualitativo sugiere que el precio ya descuenta crecimiento sostenido de FCF, estabilidad de margenes y continuidad de recompras. FactSet proyecta FCF de USD 139.7bn en FY2026 y USD 163.6bn en FY2028; incluso si se materializa, el activo no luce barato porque parte de ese crecimiento ya esta incorporado. El precio objetivo medio de USD 307.66 implica solo 4.9% de retorno esperado."))
story.append(table([
    ["Multiplo", "FY21", "FY22", "FY23", "FY24", "FY25"],
    ["P/S", "6.8x", "6.2x", "7.1x", "9.0x", "9.2x"],
    ["P/E", "26.2x", "24.6x", "27.9x", "37.4x", "34.2x"],
    ["P/FCF", "26.7x", "22.0x", "27.2x", "32.3x", "38.8x"],
    ["EV/Sales", "6.6x", "5.8x", "7.1x", "9.1x", "9.2x"],
    ["EV/EBITDA", "19.9x", "17.5x", "21.7x", "26.6x", "26.4x"],
], [1.7 * inch, 0.88 * inch, 0.88 * inch, 0.88 * inch, 0.88 * inch, 0.88 * inch]))
story.append(p("El credito es muy favorable: S&P AA+ estable, deuda total de USD 84.7bn a marzo de 2026, caja e inversiones de corto plazo de USD 68.5bn, net debt/EBITDA de 0.1x, total debt/EBITDA de 0.5x y Altman Z-Score de 11.1. No hay senal de estres crediticio; el problema Value es precio, no solvencia."))

story.append(p("Risks & Thesis Breakers", "H1G"))
for bullet in [
    "<b>Valuacion premium:</b> compresion de multiples si tasas reales suben, si AI decepciona o si el crecimiento de iPhone se normaliza.",
    "<b>Greater China:</b> exposicion a demanda, regulacion, competencia local y tensiones geopoliticas.",
    "<b>Ciclo de hardware:</b> dependencia de iPhone y reposicion de dispositivos.",
    "<b>Regulacion:</b> App Store, comisiones, pagos, privacidad, competencia y DMA/antitrust.",
    "<b>Capital allocation:</b> recompras a multiplos altos pueden sostener EPS pero reducir creacion marginal de valor.",
]:
    story.append(p("• " + bullet))
story.append(p("La tesis Value se invalidaria si el precio siguiera expandiendose sin crecimiento proporcional de FCF, si el FCF yield cae por debajo de 2.5%, si Services pierde traccion o margen, si China acelera deterioro, o si la compania aumenta recompras a precios que destruyen valor intrinseco por accion."))

story.append(p("Portfolio Decision", "H1G"))
score_data = [
    ["Bloque", "Peso", "Score", "Lectura"],
    ["Descuento relativo", "15", "2", "Multiplo alto vs historia y pares; no hay descuento estadistico."],
    ["Margen de seguridad", "20", "3", "Upside de consenso de 4.9%, inferior al minimo Value de 20%."],
    ["Calidad de flujo y utilidades", "15", "14", "FCF positivo, recurrente y conversion alta de utilidad a caja."],
    ["Balance y solvencia", "15", "15", "AA+, net debt/EBITDA de 0.1x y liquidez amplia."],
    ["Rentabilidad y ROIC", "10", "10", "ROIC de 70.6% frente a WACC de 9.05%."],
    ["Posicion competitiva", "10", "10", "Moat global, marca, ecosistema y pricing power."],
    ["Gobierno y capital allocation", "5", "4", "Dividendos y recompras; penaliza precio alto de recompra."],
    ["Catalizadores", "5", "1", "No hay catalizador claro de rerating Value."],
    ["Macro-liquidez", "5", "3", "Liquidez funcional, pero tasas reales altas exigen descuento."],
    ["Total", "100", "62", "Value condicionado / Watchlist."],
]
story.append(table(score_data, [1.65 * inch, 0.48 * inch, 0.55 * inch, 3.55 * inch], 8.7))
story.append(p("<b>Recomendacion:</b> Apple debe permanecer en watchlist para una estrategia Value. Puede formar parte de un portafolio por calidad, liquidez, baja probabilidad de deterioro financiero y defensa ante desaceleracion ordenada, pero no cumple hoy los requisitos de Value invertible. Para un mandato estrictamente Value, la entrada deberia esperar una correccion de precio, una mejora sustancial del FCF forward o una evidencia de catalizador que eleve valor intrinseco con margen de seguridad."))

doc.build(story)
print(OUT)
