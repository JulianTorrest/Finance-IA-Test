"""Genera un PDF de presentación a partir de docs/PRESENTACION.md."""
import re

from fpdf import FPDF


def _clean_markdown(text):
    text = text.strip()
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`{3}.*\n", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _split_slides(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Separa por encabezados ## (cada slide)
    parts = re.split(r"\n##\s+", content)
    slides = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = _clean_markdown(part)
        slides.append(part)
    return slides


def build_pdf(input_path="docs/PRESENTACION.md", output_path="docs/PRESENTACION.pdf"):
    slides = _split_slides(input_path)

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    # Usa fuentes del sistema Windows para soporte Unicode
    pdf.add_font("ArialUnicode", "", r"C:\Windows\Fonts\arial.ttf", uni=True)
    pdf.add_font("ArialUnicode", "B", r"C:\Windows\Fonts\arialbd.ttf", uni=True)

    for i, slide in enumerate(slides, start=1):
        pdf.add_page()
        pdf.set_fill_color(245, 247, 250)
        pdf.rect(0, 0, 297, 210, "F")

        # Título
        pdf.set_font("ArialUnicode", "B", 22)
        pdf.set_text_color(33, 37, 41)
        # La primera línea es el título
        first_line = slide.splitlines()[0] if slide else f"Diapositiva {i}"
        pdf.cell(0, 20, first_line, ln=True, align="C")
        pdf.ln(5)

        # Cuerpo
        body = "\n".join(slide.splitlines()[1:]).strip()
        pdf.set_font("ArialUnicode", "", 13)
        pdf.set_text_color(52, 58, 64)

        # Inserta texto con saltos de línea controlados
        for line in body.splitlines():
            if not line.strip():
                pdf.ln(4)
                continue
            # Tablas o bloques de código se tratan como texto normal
            pdf.multi_cell(0, 8, line, align="L")
            pdf.ln(2)

    pdf.output(output_path)
    print(f"PDF generado: {output_path}")


if __name__ == "__main__":
    build_pdf()
