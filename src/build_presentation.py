"""Genera un PDF de presentación a partir de docs/PRESENTACION.md."""
import os
import re

from fpdf import FPDF

import build_presentation_assets as assets


def _split_slides(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = re.split(r"\n---\s*\n", content)
    slides = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        slides.append("\n".join(lines))
    return slides


def _strip_formatting(line):
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"^\s*[-*]\s+", "", line)
    line = re.sub(r"^\s*\|\s*", "", line)
    line = re.sub(r"\s*\|\s*$", "", line)
    return line.strip()


def _is_table_row(line):
    return line.strip().startswith("|") and line.strip().endswith("|")


def _is_separator_row(line):
    return re.match(r"^\|(\s*[-:]\s*\|)+$", line.strip()) is not None


class PresentationPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)
        self.add_font("ArialUnicode", "", r"C:\Windows\Fonts\arial.ttf", uni=True)
        self.add_font("ArialUnicode", "B", r"C:\Windows\Fonts\arialbd.ttf", uni=True)

    def title_slide(self, title, subtitle=""):
        self.add_page()
        self.set_fill_color(0, 51, 102)
        self.rect(0, 0, 297, 210, "F")
        # Círculo decorativo
        self.set_fill_color(255, 255, 255)
        self.ellipse(230, 20, 80, 80)
        self.set_text_color(255, 255, 255)
        self.set_font("ArialUnicode", "B", 46)
        self.set_y(70)
        self.cell(0, 25, title, align="C")
        self.ln(15)
        if subtitle:
            self.set_font("ArialUnicode", "", 22)
            self.set_text_color(200, 220, 240)
            self.cell(0, 15, subtitle, align="C")

    def content_slide(self, lines, image_path=None):
        self.add_page()
        # Barra superior decorativa
        self.set_fill_color(0, 51, 102)
        self.rect(0, 0, 297, 18, "F")

        title = "Diapositiva"
        if lines:
            first = lines[0].strip()
            if first.startswith("## "):
                title = _strip_formatting(first.replace("## ", ""))
                lines = lines[1:]

        # Título
        self.set_xy(15, 30)
        self.set_font("ArialUnicode", "B", 26)
        self.set_text_color(0, 51, 102)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

        # Imagen si existe
        if image_path and os.path.exists(image_path):
            y_before = self.get_y()
            self.image(image_path, x=15, y=y_before, w=180)
            self.set_xy(205, y_before)
            content_y = y_before
        else:
            self.set_x(15)
            content_y = self.get_y()

        self.set_xy(15, self.get_y() + (95 if image_path and os.path.exists(image_path) else 0))
        self.set_font("ArialUnicode", "", 13)
        self.set_text_color(40, 40, 40)

        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue

            if _is_table_row(line):
                table_lines = []
                while i < len(lines) and _is_table_row(lines[i]):
                    if not _is_separator_row(lines[i]):
                        table_lines.append(lines[i])
                    i += 1
                if table_lines:
                    self._draw_table(table_lines)
                continue

            if line.strip().startswith("```"):
                i += 1
                code_lines = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                self._draw_code_box(code_lines)
                i += 1
                continue

            if line.strip().startswith("- ") or line.strip().startswith("* "):
                self._bullet(line)
            else:
                self._body_text(line)
            i += 1
            self.ln(2)

    def _body_text(self, line):
        self.set_font("ArialUnicode", "", 13)
        self.set_text_color(40, 40, 40)
        self.multi_cell(260, 7, _strip_formatting(line), align="L")

    def _bullet(self, line):
        self.set_x(20)
        self.set_font("ArialUnicode", "B", 13)
        self.set_text_color(0, 51, 102)
        self.cell(8, 7, "-", new_x="RIGHT", new_y="TOP")
        self.set_font("ArialUnicode", "", 13)
        self.set_text_color(40, 40, 40)
        self.multi_cell(250, 7, _strip_formatting(line), align="L")

    def _draw_table(self, table_lines):
        rows = [[_strip_formatting(cell).strip() for cell in line.split("|") if cell.strip() != ""] for line in table_lines]
        if not rows:
            return
        cols = len(rows[0])
        col_w = 260 / cols
        start_x = 15
        start_y = self.get_y()
        row_h = 9

        for r_idx, row in enumerate(rows):
            x = start_x
            for cell in row:
                if r_idx == 0:
                    self.set_fill_color(0, 51, 102)
                    self.set_text_color(255, 255, 255)
                    self.set_font("ArialUnicode", "B", 10)
                    self.rect(x, start_y, col_w, row_h, "F")
                    self.set_xy(x, start_y + 2)
                    self.cell(col_w, 6, cell, align="C")
                else:
                    self.set_fill_color(245, 247, 250)
                    self.set_text_color(40, 40, 40)
                    self.set_font("ArialUnicode", "", 10)
                    self.rect(x, start_y, col_w, row_h, "F")
                    self.set_xy(x + 2, start_y + 2)
                    self.cell(col_w - 4, 6, cell, align="L")
                x += col_w
            start_y += row_h
            if start_y > 190:
                self.add_page()
                start_y = 30
                start_x = 15
        self.set_y(start_y + 5)

    def _draw_code_box(self, code_lines):
        self.set_fill_color(230, 230, 230)
        start_y = self.get_y()
        h = max(len(code_lines) * 6 + 6, 12)
        self.rect(20, start_y, 260, h, "F")
        self.set_xy(25, start_y + 4)
        self.set_font("ArialUnicode", "", 11)
        self.set_text_color(40, 40, 40)
        for code in code_lines:
            self.set_x(25)
            self.cell(0, 6, code, new_x="LMARGIN", new_y="NEXT")
        self.set_y(start_y + h + 4)


def _image_for_title(title):
    mapping = {
        "Arquitectura": "docs/imgs/arquitectura.png",
        "Cronograma": "docs/imgs/cronograma.png",
        "Modelos desarrollados": "docs/imgs/metricas_riesgo.png",
        "Resultados": "docs/imgs/metricas_riesgo.png",
        "MLOps implementado": "docs/imgs/mlops_checklist.png",
    }
    for key, path in mapping.items():
        if key.lower() in title.lower():
            return path
    return None


def build_pdf(input_path="docs/PRESENTACION.md", output_path="docs/PRESENTACION.pdf"):
    # Genera assets visuales primero
    assets.build_all()

    raw_slides = _split_slides(input_path)
    pdf = PresentationPDF()

    # Portada
    pdf.title_slide("Prueba Tecnica", "Data Scientist Senior - Banca Analitica")

    for slide in raw_slides:
        lines = slide.splitlines()
        title_line = lines[0].strip() if lines else ""
        title = _strip_formatting(title_line.replace("## ", "")) if title_line.startswith("## ") else ""
        image = _image_for_title(title)
        pdf.content_slide(lines, image_path=image)

    pdf.output(output_path)
    print(f"PDF generado: {output_path}")


if __name__ == "__main__":
    build_pdf()
