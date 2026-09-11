from __future__ import annotations

import pytest

from src.product_s2g.config import ProductConfig
from src.product_s2g.corpus import FullCorpusCatalog
from src.product_s2g.streamlit_app import document_rows, page_label


@pytest.fixture(scope="module")
def catalog():
    return FullCorpusCatalog.load(ProductConfig.load())


def test_page_label_handles_single_and_multiple_pages():
    assert page_label(3, 3) == "trang 3"
    assert page_label(3, 4) == "trang 3–4"


def test_document_rows_cover_all_source_documents(catalog):
    rows = document_rows(catalog)
    assert len(rows) == 27
    assert sum(row["chunks"] for row in rows) == 3482
    assert all(row["chunks"] > 0 for row in rows)
    assert len({row["document_id"] for row in rows}) == 27
