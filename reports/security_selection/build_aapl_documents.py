from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


NAVY = "1E3D70"
PALE = "D8DEE7"


def set_font(run, size=12, bold=False, italic=False, color="202428"):
    run.font.name = "Garamond"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Garamond")
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_p(doc, text, style=None, bold=False):
    p = doc.add_paragraph(style=style)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_font(run, 12, bold=bold)
    return p


def add_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(value))
            set_font(r, 9, bold=(i == 0), color=("FFFFFF" if i == 0 else "202428"))
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i == 0:
                shade(cell, NAVY)
    doc.add_paragraph()
    return table


def setup_doc(title):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.82)
    section.bottom_margin = Inches(0.82)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    styles = doc.styles
    styles["Normal"].font.name = "Garamond"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Garamond")
    styles["Normal"].font.size = Pt(12)
    for name, size in [("Title", 22), ("Heading 1", 16), ("Heading 2", 13)]:
        st = styles[name]
        st.font.name = "Garamond"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Garamond")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(NAVY)
    h = section.header.paragraphs[0]
    h.text = title
    set_font(h.runs[0], 9, color="7E93B0")
    return doc


def build_main():
    doc = setup_doc("AAPL | Value Investment Memo")
    p = doc.add_paragraph()
    r = p.add_run("Apple Inc. (AAPL)\nValue Investment Memo")
    set_font(r, 22, bold=True, color=NAVY)
    add_p(doc, "10 de mayo de 2026 | NASDAQ: AAPL | Filosofia: Value")
    doc.add_page_break()
    doc.add_paragraph("Contents", style="Heading 1")
    for i, title in enumerate([
        "Executive Summary",
        "Investment View",
        "Business Model & Cash Engine",
        "Competitive Position & Addressable Market",
        "Macro & Liquidity Backdrop",
        "Financial Profile & Estimate Quality",
        "Quality of Earnings, ROIC & Capital Return",
        "Valuation, Reverse DCF & Credit",
        "Risks & Thesis Breakers",
        "Portfolio Decision",
    ], 1):
        add_p(doc, f"{i}. {title}")
    doc.add_page_break()

    sections = [
        ("Executive Summary", "Apple no clasifica hoy como un activo Value invertible en sentido estricto. La compania presenta una calidad financiera extraordinaria: flujo libre recurrente, retorno sobre capital invertido superior al costo de capital, balance resiliente, marca global, ecosistema cerrado y capacidad de recompras a escala. Sin embargo, el precio actual no ofrece el margen de seguridad minimo que exige una filosofia Value disciplinada."),
        ("Investment View", "Apple cotiza cerca de su maximo de 52 semanas y el consenso FactSet implica solo 4.9% de retorno al precio objetivo medio. El momentum fundamental no esta roto; lo que esta en duda es el precio de entrada."),
        ("Business Model & Cash Engine", "Apple monetiza un ecosistema integrado de hardware, software y servicios. iPhone produce escala y base instalada; Services mejora recurrencia, margen bruto y estabilidad."),
        ("Competitive Position & Addressable Market", "El moat es profundo por marca, base instalada, integracion, chips propios, privacidad, retail y developer ecosystem. Bajo Value, este moat justifica una prima, pero no elimina la necesidad de margen de seguridad."),
        ("Macro & Liquidity Backdrop", "El entorno favorece empresas con FCF actual, balance fuerte y pricing power. Pero tasas reales cercanas a 2% reducen la tolerancia a pagar 30x-35x utilidades por crecimiento moderado."),
        ("Financial Profile & Estimate Quality", "En FY2025, Apple genero USD 416.2bn de ingresos, USD 133.1bn de EBIT y USD 98.8bn de FCF. El consenso proyecta USD 545.0bn de ingresos en FY2028."),
        ("Quality of Earnings, ROIC & Capital Return", "El ROIC FY2025 fue 70.6% frente a WACC de 9.05%. La recompra reduce acciones, pero recomprar a multiplos altos crea menos valor marginal que recomprar con descuento."),
        ("Valuation, Reverse DCF & Credit", "La valuacion es el punto debil: P/E LTM 35.5x, EV/EBITDA 26.9x y FCF yield aproximado de 3.0%. El credito es muy favorable: S&P AA+, net debt/EBITDA de 0.1x y Altman Z-Score de 11.1."),
        ("Risks & Thesis Breakers", "Los riesgos principales son valuacion premium, Greater China, ciclo de hardware, regulacion y recompras a multiplos altos."),
        ("Portfolio Decision", "Apple debe permanecer en watchlist para una estrategia Value. Puede formar parte de un portafolio por calidad y resiliencia, pero no cumple hoy los requisitos de Value invertible."),
    ]
    for title, body in sections:
        doc.add_paragraph(title, style="Heading 1")
        add_p(doc, body)
        if title == "Executive Summary":
            add_table(doc, [
                ["Metric", "Value", "Interpretation"],
                ["Market Cap", "USD 4.31tn", "Large-cap compounder priced at premium"],
                ["P/E LTM", "35.5x", "High for Value entry"],
                ["FCF LTM", "USD 129.2bn", "Strong cash generation"],
                ["Score", "62/100", "Value condicionado / Watchlist"],
            ])
        elif title == "Business Model & Cash Engine":
            add_table(doc, [
                ["Unidad", "% ingresos FY25", "Ingresos USD m", "Crec. YoY"],
                ["iPhone", "50.4%", "209,586", "4.2%"],
                ["Services", "26.2%", "109,158", "13.5%"],
                ["Wearables, Home & Accessories", "8.6%", "35,686", "-3.6%"],
                ["Mac", "8.1%", "33,708", "12.4%"],
                ["iPad", "6.7%", "28,023", "5.0%"],
            ])
        elif title == "Competitive Position & Addressable Market":
            add_table(doc, [
                ["Region", "% ingresos FY25", "Ingresos USD m", "Crec. YoY"],
                ["United States", "36.5%", "151,790", "6.7%"],
                ["Europe", "26.7%", "111,032", "9.6%"],
                ["Greater China", "15.5%", "64,377", "-3.8%"],
                ["Rest of Asia Pacific", "8.1%", "33,696", "9.9%"],
                ["Japan", "6.9%", "28,703", "14.6%"],
            ])
        elif title == "Financial Profile & Estimate Quality":
            add_table(doc, [
                ["Metrica", "FY23", "FY24", "FY25", "FY26E", "FY27E", "FY28E"],
                ["Ingresos (USD m)", "383,285", "391,035", "416,161", "474,033", "512,521", "544,972"],
                ["EBIT (USD m)", "114,301", "123,216", "133,050", "153,749", "164,701", "178,670"],
                ["Net income (USD m)", "96,995", "93,736", "112,010", "127,377", "137,235", "149,203"],
                ["EPS diluido", "6.13", "6.08", "7.47", "8.69", "9.55", "10.53"],
                ["FCF (USD m)", "99,584", "108,807", "98,767", "139,674", "148,874", "163,636"],
            ])
        elif title == "Quality of Earnings, ROIC & Capital Return":
            add_table(doc, [
                ["Ratio", "FY21", "FY22", "FY23", "FY24", "FY25"],
                ["Gross margin", "41.8%", "43.3%", "44.1%", "46.2%", "46.9%"],
                ["Operating margin", "29.8%", "30.3%", "29.8%", "31.5%", "32.0%"],
                ["Net margin", "25.9%", "25.3%", "25.3%", "24.0%", "26.9%"],
                ["FCF margin", "25.4%", "28.3%", "26.0%", "27.8%", "23.7%"],
                ["ROIC", "53.4%", "58.2%", "59.0%", "58.2%", "70.6%"],
            ])
        elif title == "Valuation, Reverse DCF & Credit":
            add_table(doc, [
                ["Multiplo", "FY21", "FY22", "FY23", "FY24", "FY25"],
                ["P/S", "6.8x", "6.2x", "7.1x", "9.0x", "9.2x"],
                ["P/E", "26.2x", "24.6x", "27.9x", "37.4x", "34.2x"],
                ["P/FCF", "26.7x", "22.0x", "27.2x", "32.3x", "38.8x"],
                ["EV/Sales", "6.6x", "5.8x", "7.1x", "9.1x", "9.2x"],
                ["EV/EBITDA", "19.9x", "17.5x", "21.7x", "26.6x", "26.4x"],
            ])
        elif title == "Portfolio Decision":
            add_table(doc, [
                ["Bloque", "Peso", "Score", "Lectura"],
                ["Descuento relativo", "15", "2", "Multiplo alto vs historia y pares."],
                ["Margen de seguridad", "20", "3", "Upside de consenso inferior al umbral Value."],
                ["Calidad de flujo y utilidades", "15", "14", "FCF positivo y recurrente."],
                ["Balance y solvencia", "15", "15", "AA+, net debt/EBITDA bajo."],
                ["Rentabilidad y ROIC", "10", "10", "ROIC superior al WACC."],
                ["Total", "100", "62", "Value condicionado / Watchlist."],
            ])
    doc.save("AAPL_value_investment_memo.docx")


def build_sources():
    doc = setup_doc("AAPL | Source Notes")
    doc.add_paragraph("AAPL Value Memo - Source Notes / Audit Trail", style="Title")
    add_p(doc, "Este documento funciona como soporte auditable del memo de Apple. Cada fila vincula una pieza material de informacion con el archivo, hoja, seccion o fuente publica utilizada.")
    rows = [
        ["Dato / afirmacion", "Fuente precisa", "Ubicacion auditable"],
        ["Metodologia Value", "01.Metodologia Value.pdf", "Etapa 10: score final y dictamen."],
        ["Valuation snapshot", "AAPL.zip > Overview/Snapshot_AAPL-US_2026-05-10T12_31_32_AppleInc.xlsx", "Sheet AAPL-US; rows 24-32."],
        ["Target/rating", "AAPL.zip > Overview/Snapshot_AAPL-US_2026-05-10T12_31_32_AppleInc.xlsx", "Sheet AAPL-US; rows 36-41."],
        ["Financial Summary", "AAPL.zip > Overview/Snapshot_AAPL-US_2026-05-10T12_31_32_AppleInc.xlsx", "Rows 92-108."],
        ["Business Unit Revenue", "AAPL.zip > Overview/Snapshot_AAPL-US_2026-05-10T12_31_32_AppleInc.xlsx", "Rows 142-150."],
        ["Geographic Revenue", "AAPL.zip > Overview/Snapshot_AAPL-US_2026-05-10T12_31_32_AppleInc.xlsx", "Rows 152-160."],
        ["Margins and ROIC", "AAPL.zip > Financial/ratio_analysis_20260510_20FQ.xlsx", "Rows 10-24 and 25-36."],
        ["Credit", "AAPL.zip > Credit_Analysis/20260510_122835576PM_DCS Overview_AAPL-US.xlsx", "Rows 4-8, 11-20, 22-49."],
        ["Q2 FY2026", "Apple Newsroom", "Apple reports second quarter results, 30-Apr-2026."],
        ["Corporate description and risks", "Apple FY2025 Form 10-K, SEC EDGAR", "Business, Products and Services, Risk Factors."],
        ["Macro data", "BLS, Federal Reserve H.4.1, FRED", "CPI, H.4.1, NFCI, DFII10, T10YIE."],
    ]
    add_table(doc, rows)
    doc.save("AAPL_value_source_notes.docx")


if __name__ == "__main__":
    build_main()
    build_sources()
