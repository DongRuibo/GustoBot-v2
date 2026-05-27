"""文件解析测试。"""

import base64
from io import BytesIO
import zipfile

import pytest

from app.files.parser import FileParser, UnsupportedFileTypeError


def test_csv_base64_parser_extracts_text_and_schema() -> None:
    payload = "菜名,菜系,热度\n宫保鸡丁,川菜,96\n佛跳墙,闽菜,91\n".encode("utf-8")

    parsed = FileParser().parse(
        {
            "type": "file",
            "filename": "菜谱统计.csv",
            "content_base64": base64.b64encode(payload).decode("ascii"),
        }
    )

    assert parsed is not None
    assert parsed.parser_type == "csv"
    assert "宫保鸡丁" in parsed.content
    assert parsed.metadata["schema_catalog"]["columns"][0]["name"] == "菜名"
    assert parsed.metadata["schema_catalog"]["row_count"] == 2


def test_xlsx_base64_parser_extracts_text_and_schema() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    Workbook = openpyxl.Workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "菜谱"
    sheet.append(["菜名", "菜系", "热度"])
    sheet.append(["宫保鸡丁", "川菜", 96])
    sheet.append(["佛跳墙", "闽菜", 91])
    buffer = BytesIO()
    workbook.save(buffer)

    parsed = FileParser().parse(
        {
            "type": "file",
            "filename": "菜谱统计.xlsx",
            "content_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    )

    assert parsed is not None
    assert parsed.parser_type == "xlsx"
    assert "宫保鸡丁" in parsed.content
    assert parsed.metadata["schema_catalog"]["sheets"][0]["columns"][0]["name"] == "菜名"
    assert parsed.metadata["schema_catalog"]["sheets"][0]["row_count"] == 2


def test_docx_parser_extracts_paragraphs_and_tables() -> None:
    payload = _build_docx_bytes()

    parsed = FileParser().parse(
        {
            "type": "file",
            "filename": "宫保鸡丁说明.docx",
            "content_base64": base64.b64encode(payload).decode("ascii"),
        }
    )

    assert parsed is not None
    assert parsed.parser_type == "docx"
    assert "制作要点" in parsed.content
    assert "食材 | 用量" in parsed.content
    assert parsed.metadata["paragraph_count"] == 1
    assert parsed.metadata["table_count"] == 1


def test_pdf_parser_extracts_all_pages_when_pypdf_available() -> None:
    pytest.importorskip("pypdf")
    payload = _build_simple_pdf("Gongbao chicken knowledge")

    parsed = FileParser().parse(
        {
            "type": "file",
            "filename": "gongbao.pdf",
            "content_base64": base64.b64encode(payload).decode("ascii"),
        }
    )

    assert parsed is not None
    assert parsed.parser_type == "pdf"
    assert "Gongbao chicken knowledge" in parsed.content
    assert parsed.metadata["page_count"] == 1
    assert parsed.metadata["parser_backend"] == "pypdf"


def test_legacy_office_formats_are_unsupported() -> None:
    parser = FileParser()

    with pytest.raises(UnsupportedFileTypeError, match="unsupported_file_type"):
        parser.parse({"type": "file", "filename": "old.xls", "content_base64": base64.b64encode(b"fake").decode("ascii")})

    with pytest.raises(UnsupportedFileTypeError, match="unsupported_file_type"):
        parser.parse({"type": "file", "filename": "old.doc", "content_base64": base64.b64encode(b"fake").decode("ascii")})


def _build_docx_bytes() -> bytes:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>制作要点</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>食材</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>用量</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>鸡肉</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>300g</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def _build_simple_pdf(text: str) -> bytes:
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)
