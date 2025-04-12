import mimetypes
from pathlib import Path

import re
import fitz # PyMuPDF
import docx
import openpyxl
import pandas as pd
import odf.text
import odf.opendocument
from striprtf.striprtf import rtf_to_text
import xml.etree.ElementTree as ET
from ocr import OCR


class DocExtractor:
    ocr = OCR()

    @staticmethod
    def extract_paragraphs(file_path): # -> [(page_num, paragraph), (None, paragraph)...]
        file_path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)

        handler = DocExtractor._get_handlers().get(mime_type)
        if handler:
            try:
                return handler(file_path)
            except Exception as e:
                print(f"Primary handler failed for {mime_type}: {e}")
                if mime_type == "application/pdf":
                    return DocExtractor._handle_pdf_ocr(file_path)
                raise e
        else:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

    @staticmethod
    def extract_text(file_path):
        return '\n\n'.join(paragraph for _, paragraph in DocExtractor.extract_paragraphs(file_path))

    @staticmethod
    def _get_handlers():
        return {
            'application/pdf': DocExtractor._handle_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocExtractor._handle_docx,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DocExtractor._handle_xlsx,
            'application/vnd.oasis.opendocument.text': DocExtractor._handle_odt,
            'application/rtf': DocExtractor._handle_rtf,
            'text/plain': DocExtractor._handle_txt,
            'text/csv': DocExtractor._handle_csv,
            'text/xml': DocExtractor._handle_xml,
            'application/xml': DocExtractor._handle_xml,
        }

    # === Handlers ===

    @staticmethod
    def _handle_pdf(file_path):
        doc = fitz.open(file_path)
        paragraphs = []
        for i, page in enumerate(doc):
            text = page.get_text()
            page_paragraphs = DocExtractor._split_into_paragraphs(text)
            paragraphs.extend((i + 1, p) for p in page_paragraphs)
        if not paragraphs:
            raise ValueError("PDF text extraction failed; fallback to OCR")
        return paragraphs

    @staticmethod
    def _handle_pdf_ocr(file_path):
        doc_data = DocExtractor.ocr.extract_pdf_data(file_path)
        paragraphs = []
        for i, page in enumerate(doc_data.pages):
            for block in page.blocks:
                text = OCR._convert_block_data_to_text(block)
                if text.strip():
                    paragraphs.append((i + 1, text.strip()))
        return paragraphs

    @staticmethod
    def _handle_docx(file_path):
        doc = docx.Document(file_path)
        return [(None, p.text.strip()) for p in doc.paragraphs if p.text.strip()]

    @staticmethod
    def _handle_xlsx(file_path):
        wb = openpyxl.load_workbook(file_path, data_only=True)
        paragraphs = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                values = [str(cell) for cell in row if cell is not None]
                if values:
                    paragraphs.append((None, " | ".join(values)))
        return paragraphs

    @staticmethod
    def _handle_odt(file_path):
        textdoc = odf.opendocument.load(str(file_path))
        text_elements = textdoc.getElementsByType(odf.text.P)
        return [(None, str(p).strip()) for p in text_elements if str(p).strip()]

    @staticmethod
    def _handle_rtf(file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            text = rtf_to_text(file.read())
            return [(None, p) for p in DocExtractor._split_into_paragraphs(text)]

    @staticmethod
    def _handle_txt(file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            return [(None, p) for p in DocExtractor._split_into_paragraphs(text)]

    @staticmethod
    def _handle_csv(file_path):
        df = pd.read_csv(file_path)
        lines = [line for line in df.to_string(index=False).split("\n") if line.strip()]
        return [(None, line) for line in lines]

    @staticmethod
    def _handle_xml(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        text = DocExtractor._extract_xml_text(root)
        return [(None, p) for p in DocExtractor._split_into_paragraphs(text)]

    @staticmethod
    def _extract_xml_text(element):
        texts = []

        if element.text and element.text.strip():
            texts.append(element.text.strip())

        for child in element:
            texts.append(DocExtractor._extract_xml_text(child))

        if element.tail and element.tail.strip():
            texts.append(element.tail.strip())

        return " ".join(texts)

    @staticmethod
    def _split_into_paragraphs(text):
        # Splits on two or more newline-separated whitespace-only lines
        return [p.strip() for p in re.split(r'(?:\r?\n\s*){2,}', text) if p.strip()]