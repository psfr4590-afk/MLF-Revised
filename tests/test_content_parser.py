from __future__ import annotations

import json

import pytest

from pipeline.crawler.content_parser import ContentParseError, parse_content


def test_csv_preserves_quoted_fields_and_headers() -> None:
    data = b'name,description,value\n"alpha, beta","quoted value",42\nsecond,plain,7\n'
    parsed = parse_content("https://example.test/data.csv", "text/csv", data)
    assert parsed.content_type == "tabular_data"
    assert "name: alpha, beta" in parsed.text
    assert "description: quoted value" in parsed.text
    assert "value: 42" in parsed.text
    assert parsed.metadata["rows"] == 2


def test_tsv_uses_tab_delimiter() -> None:
    data = b"name\tscore\nalpha\t0.5\nbeta\t0.9\n"
    parsed = parse_content("https://example.test/data.tsv", "text/tab-separated-values", data)
    assert "name: alpha" in parsed.text
    assert "score: 0.9" in parsed.text
    assert parsed.metadata["delimiter"] == "\t"


def test_json_array_is_record_preserving() -> None:
    data = json.dumps([{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]).encode()
    parsed = parse_content("https://example.test/data.json", "application/json", data)
    assert parsed.content_type == "structured_data"
    assert '"id":1' in parsed.text
    assert '"text":"second"' in parsed.text
    assert parsed.metadata["records"] == 2


def test_jsonl_reports_bad_line_instead_of_silently_dropping_it() -> None:
    data = b'{"id":1}\nnot-json\n'
    with pytest.raises(ContentParseError, match="line 2"):
        parse_content("https://example.test/data.jsonl", "application/json", data)


def test_xml_preserves_element_text() -> None:
    data = b"<root><title>Example</title><value>42</value></root>"
    parsed = parse_content("https://example.test/data.xml", "application/xml", data)
    assert parsed.content_type == "structured_data"
    assert "Example" in parsed.text
    assert "42" in parsed.text


def test_html_preserves_title_and_links() -> None:
    data = b'<html><head><title>Example</title></head><body><p>Hello</p><a href="/next">Next</a></body></html>'
    parsed = parse_content("https://example.test/index.html", "text/html", data)
    assert parsed.title == "Example"
    assert parsed.text == "Example Hello"
    assert parsed.links == ["/next"]


def test_unknown_binary_is_fail_closed() -> None:
    with pytest.raises(ContentParseError, match="unsupported content type"):
        parse_content("https://example.test/blob.bin", "application/octet-stream", b"\\x00\\x01\\x02")


def test_image_only_pdf_is_not_ingested_as_empty_document() -> None:
    # A minimal valid PDF with no text. The parser must reject it rather than
    # emitting an empty training document.
    from pypdf import PdfWriter
    from io import BytesIO

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ContentParseError, match="image-only PDFs"):
        parse_content("https://example.test/scan.pdf", "application/pdf", output.getvalue())
