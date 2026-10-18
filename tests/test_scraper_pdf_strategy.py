"""PDF extraction tries the text layer before layout models and OCR."""

import pytest

from src.scraper_service import scraper as scraper_module
from src.scraper_service.scraper import ScraperService


try:
    from unstructured.documents.elements import NarrativeText
except ImportError:  # pragma: no cover - unstructured is a runtime dependency
    NarrativeText = None


pytestmark = pytest.mark.skipif(NarrativeText is None, reason="unstructured missing")


def _record_strategies(monkeypatch, results: dict[str, object]) -> list[str]:
    """Replaces partition_pdf; returns the list of strategies it was called with."""
    calls: list[str] = []

    def fake_partition_pdf(filename, strategy, **kwargs):
        calls.append(strategy)
        result = results.get(strategy, [])
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(scraper_module, "partition_pdf", fake_partition_pdf)
    return calls


def test_text_pdf_uses_fast_text_layer_only(monkeypatch, tmp_path):
    pdf = tmp_path / "flamingos.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    calls = _record_strategies(
        monkeypatch, {"fast": [NarrativeText("Flamingos are pink wading birds.")]}
    )

    elements = ScraperService().extract_elements(str(pdf))

    assert calls == ["fast"]
    assert [e.text for e in elements] == ["Flamingos are pink wading birds."]


def test_pdf_without_text_layer_falls_back_to_layout_then_ocr(monkeypatch, tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    calls = _record_strategies(
        monkeypatch,
        {
            "fast": [NarrativeText("   ")],
            "hi_res": RuntimeError("tesseract is not installed"),
            "ocr_only": [NarrativeText("Scanned text")],
        },
    )

    elements = ScraperService().extract_elements(str(pdf))

    assert calls == ["fast", "hi_res", "ocr_only"]
    assert [e.text for e in elements] == ["Scanned text"]


def test_fast_strategy_error_still_tries_the_other_strategies(monkeypatch, tmp_path):
    pdf = tmp_path / "odd.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    calls = _record_strategies(
        monkeypatch,
        {
            "fast": RuntimeError("pdfminer failed"),
            "hi_res": [NarrativeText("Layout text")],
        },
    )

    elements = ScraperService().extract_elements(str(pdf))

    assert calls == ["fast", "hi_res"]
    assert [e.text for e in elements] == ["Layout text"]
