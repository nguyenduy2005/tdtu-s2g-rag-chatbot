"""Streamlit user interface for the full-corpus adapted S2G-RAG product."""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product_s2g.config import ProductConfig  # noqa: E402
from src.product_s2g.corpus import FullCorpusCatalog  # noqa: E402
from src.product_s2g.service import ProductRuntimeError, S2GChatService  # noqa: E402


EXAMPLE_QUESTIONS = (
    "Khi nào sinh viên bị hạ bậc xếp loại tốt nghiệp?",
    "Sinh viên phải thực hiện những yêu cầu gì khi vào phòng thi?",
    "Điều kiện cấp chứng nhận kỹ sư hoặc cử nhân ưu tú là gì?",
)


@st.cache_resource(show_spinner=False)
def load_catalog() -> FullCorpusCatalog:
    """Load and integrity-check the immutable corpus without loading ML models."""

    return FullCorpusCatalog.load(ProductConfig.load())


@st.cache_resource(show_spinner=False)
def load_service() -> S2GChatService:
    """Build the heavy retrieval/controller stack once per Streamlit process."""

    return S2GChatService.build(ProductConfig.load())


def page_label(page_start: int, page_end: int) -> str:
    return f"trang {page_start}" if page_start == page_end else f"trang {page_start}–{page_end}"


def document_rows(catalog: FullCorpusCatalog) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    roles: defaultdict[str, Counter[str]] = defaultdict(Counter)
    titles: dict[str, str] = {}
    for row in catalog.chunks.values():
        document_id = row["document_id"]
        counts[document_id] += 1
        roles[document_id][row["retrieval_role"]] += 1
        titles[document_id] = row["document_title"]
    unavailable: defaultdict[str, list[int]] = defaultdict(list)
    for row in catalog.unavailable_pages:
        unavailable[row["document_id"]].append(row["page"])
    return [
        {
            "document_id": document_id,
            "title": titles[document_id],
            "source_file": catalog.source_files[document_id].name,
            "chunks": counts[document_id],
            "roles": dict(roles[document_id]),
            "unavailable_pages": unavailable[document_id],
        }
        for document_id in catalog.document_ids
    ]


def render_citations(response: dict[str, Any], catalog: FullCorpusCatalog, message_index: int) -> None:
    citations = response.get("citations") or []
    if not citations:
        st.info("Lượt trả lời này không có nguồn đủ điều kiện để trích dẫn.")
        return
    with st.expander(f"Nguồn kiểm chứng ({len(citations)})", expanded=True):
        for citation_index, citation in enumerate(citations, start=1):
            trust_text = "Cần kiểm tra PDF gốc" if citation["low_trust"] else "Evidence chuẩn"
            trust_icon = "⚠️" if citation["low_trust"] else "✅"
            st.markdown(
                f"**[{citation_index}] {citation['document_title']}**  \n"
                f"{page_label(citation['page_start'], citation['page_end'])} · "
                f"`{citation['retrieval_role']}` · {trust_icon} {trust_text}"
            )
            if citation.get("excerpt"):
                st.write(citation["excerpt"])
            flags = citation.get("quality_flags") or []
            if flags:
                st.caption("Quality flags: " + ", ".join(flags))
            st.caption("Chunk: " + citation["chunk_id"])
            pdf_path = catalog.source_files.get(citation["document_id"])
            if pdf_path and pdf_path.is_file():
                st.download_button(
                    "Tải PDF gốc",
                    data=lambda path=pdf_path: path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"pdf-{message_index}-{citation_index}-{citation['chunk_id']}",
                    icon=":material/download:",
                )
            if citation_index != len(citations):
                st.divider()


def render_assistant(response: dict[str, Any], catalog: FullCorpusCatalog, message_index: int) -> None:
    st.markdown(response["answer"])
    if response.get("insufficient"):
        st.warning("Evidence hiện tại chưa đủ để trả lời đầy đủ. Hãy kiểm tra nguồn hoặc hỏi cụ thể hơn.")
    retrieval = response.get("retrieval") or {}
    cols = st.columns(4)
    cols[0].metric("Độ tin cậy", str(response.get("confidence", "unknown")).upper())
    cols[1].metric("Vòng S2G", retrieval.get("rounds", 0))
    cols[2].metric("Câu evidence", retrieval.get("selected_sentence_count", 0))
    cols[3].metric("Nguồn cha", retrieval.get("final_parent_chunk_count", 0))
    st.caption(
        f"Stop reason: `{retrieval.get('stop_reason', 'unknown')}` · "
        f"Request ID: `{response.get('request_id', 'unknown')}`"
    )
    render_citations(response, catalog, message_index)


def render_sidebar(catalog: FullCorpusCatalog) -> None:
    with st.sidebar:
        st.header("Trạng thái hệ thống")
        key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        st.success("OPENAI_API_KEY đã nhận") if key_present else st.error("Chưa có OPENAI_API_KEY")
        col1, col2 = st.columns(2)
        col1.metric("PDF", len(catalog.source_files))
        col2.metric("Chunks", len(catalog.chunks))
        st.caption(
            f"{catalog.page_count} trang khai báo · "
            f"{catalog.role_counts['content']} content · "
            f"{catalog.role_counts['review']} review · "
            f"{catalog.role_counts['context_only']} context-only"
        )
        if st.button("Khởi tạo S2G engine", width="stretch", disabled=not key_present):
            with st.spinner("Đang nạp dense encoder, reranker và corpus…"):
                load_service()
            st.success("S2G engine đã sẵn sàng")
        if st.button("Xóa lịch sử chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Tài liệu trong corpus")
        rows = document_rows(catalog)
        chosen_title = st.selectbox("Chọn tài liệu", [row["title"] for row in rows])
        chosen = next(row for row in rows if row["title"] == chosen_title)
        st.write(f"**{chosen['chunks']} chunks**")
        st.caption(
            " · ".join(f"{role}: {count}" for role, count in sorted(chosen["roles"].items()))
        )
        if chosen["unavailable_pages"]:
            pages = ", ".join(str(page) for page in chosen["unavailable_pages"])
            st.warning(f"Trang không có text sử dụng được: {pages}")

        st.divider()
        st.caption(
            "Phạm vi: toàn bộ 27 PDF và mọi chunk không rỗng của Canonical Corpus V1.1. "
            "Kết quả là hỗ trợ tra cứu; quyết định chính thức phải đối chiếu PDF gốc."
        )


def main() -> None:
    st.set_page_config(
        page_title="TDTU S2G-RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 6rem;}
        [data-testid="stChatMessage"] {border: 1px solid rgba(125,125,125,.18); border-radius: 16px; padding: .35rem;}
        [data-testid="stMetric"] {background: rgba(125,125,125,.06); border-radius: 12px; padding: .55rem .75rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        catalog = load_catalog()
    except Exception as exc:
        st.error(f"Không thể xác minh corpus: {type(exc).__name__}: {exc}")
        st.stop()

    render_sidebar(catalog)
    st.title("📚 Trợ lý quy định đại học — S2G-RAG")
    st.caption("Tra cứu có dẫn nguồn trên toàn bộ nội dung đọc được của 27 PDF")
    st.info(
        "Câu trả lời chỉ được tạo từ evidence S2G đã chọn. Citation gắn nhãn ⚠️ cần được "
        "đối chiếu PDF vì có OCR hoặc thuộc vai trò review/context-only."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.subheader("Câu hỏi gợi ý")
        for example in EXAMPLE_QUESTIONS:
            st.markdown(f"- {example}")

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant(message["response"], catalog, message_index)

    key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    prompt = st.chat_input(
        "Nhập câu hỏi về quy định, học vụ, học phí, thi cử…",
        max_chars=2000,
        disabled=not key_present,
    )
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("S2G đang truy xuất, chọn evidence và kiểm tra độ đầy đủ…"):
                    response = load_service().chat(prompt)
                render_assistant(response, catalog, len(st.session_state.messages))
                st.session_state.messages.append({"role": "assistant", "response": response})
            except ProductRuntimeError as exc:
                st.error(f"S2G không thể hoàn tất: {exc}")
            except Exception as exc:
                st.error(f"Lỗi hạ tầng: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
