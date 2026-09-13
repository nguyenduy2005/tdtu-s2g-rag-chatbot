# Báo cáo chạy lại pilot S2G-RAG — 2026-09-13

## Kết luận chính

Pipeline hoàn thành kết quả hợp lệ cho 20/20 câu hỏi sau một lần chạy lại có kiểm
soát ở P004. Kết quả cho thấy retrieval cơ sở đã khá tốt ở top-10, nhưng S2G hiện
chưa tăng độ bao phủ tổng thể và còn làm mất evidence ở bước chọn/trích dẫn, đặc
biệt với câu hỏi multi-evidence. Phần sinh trả lời đúng 14/20 và đầy đủ 13/20 theo
đánh giá thủ công có hỗ trợ của agent; chỉ 11/20 câu đạt đồng thời bốn tiêu chí cốt
lõi. Đây là kết quả pilot chẩn đoán, chưa phải đánh giá độc lập bởi con người.

## Phạm vi và tính hợp lệ

- Corpus: Canonical Corpus V1.1, 27 tài liệu, 3.482 chunk.
- Pilot: 20 câu hỏi, 27 evidence groups, 23 GT chunks duy nhất.
- Trạng thái benchmark: `DRAFT_AGENT_VERIFIED_PENDING_HUMAN`.
- Runtime không nhận ground truth; evaluator offline chỉ mở GT sau khi kết quả đã
  đóng.
- Cấu hình retrieval/controller không được thay đổi trong lần chạy này.
- Kết quả mới được lưu riêng dưới
  `outputs/evaluation/s2g_pilot_rerun_20260913/`; kết quả pilot cũ không bị ghi đè.

## Chất lượng truy xuất

### B0 — Hybrid + RRF + reranker

| Chỉ số | Kết quả |
|---|---:|
| Recall@1 | 0,575 |
| Recall@3 | 0,825 |
| Recall@5 | 0,825 |
| Recall@10 | 0,825 |
| MRR@10 | 0,767 |
| Evidence-group coverage@10 | 0,825 |
| Complete-evidence@10 | 15/20 |

BM25 riêng lẻ đạt Recall/EGC@10 = 0,875 và complete-evidence@10 = 16/20, cao
hơn đầu ra sau reranker ở top-10. Reranker tăng kết quả sớm ở @1 nhưng đẩy một số
evidence đúng khỏi top-10. Đây là dấu hiệu cần kiểm tra ranking loss theo từng câu,
không phải cơ sở để tinh chỉnh trực tiếp trên chính 20 câu pilot.

### S2G theo từng giai đoạn

| Giai đoạn | Recall@1 | Recall@10 | MRR@10 | EGC@10 | Complete @10 |
|---|---:|---:|---:|---:|---:|
| Iterative candidate pool | 0,500 | 0,825 | 0,714 | 0,825 | 15/20 |
| S2G selected evidence | 0,650 | 0,800 | 0,850 | 0,800 | 14/20 |
| Answer citations | 0,600 | 0,800 | 0,817 | 0,800 | 14/20 |

Selector làm evidence đúng xuất hiện sớm hơn, thể hiện qua MRR@10 tăng, nhưng làm
giảm bao phủ từ 0,825 xuống 0,800 và giảm số câu đủ evidence từ 15 xuống 14. Vì
vậy hệ thống hiện thiên về độ chính xác sớm hơn là giữ đủ evidence.

### Theo loại câu hỏi quan trọng

| Loại | Giai đoạn | MRR@10 | EGC@10 | Complete @10 |
|---|---|---:|---:|---:|
| Multi-evidence (4) | B0 reranker | 0,625 | 0,625 | 2/4 |
| Multi-evidence (4) | S2G citations | 0,750 | 0,500 | 1/4 |
| Cross-document (3) | B0 reranker | 0,667 | 0,667 | 1/3 |
| Cross-document (3) | S2G citations | 1,000 | 0,667 | 1/3 |
| Direct lookup (5) | S2G citations | 1,000 | 1,000 | 5/5 |
| Terminology mismatch (3) | S2G citations | 1,000 | 1,000 | 3/3 |

Điểm yếu rõ nhất là multi-evidence: evidence đầu tiên thường được tìm sớm nhưng
evidence bắt buộc thứ hai bị thiếu. Cross-document cũng chỉ đủ hoàn toàn 1/3 câu,
dù evidence đầu tiên có thứ hạng tốt.

## Chất lượng thành phần sinh

Đánh giá được thực hiện bằng cách đối chiếu từng câu trả lời với câu hỏi, GT chunk
và `source_text`. Đây là đánh giá thủ công **có hỗ trợ của agent**, không phải nghiên
cứu annotation độc lập bởi người đánh giá.

| Tiêu chí | PASS | FAIL | REVIEW | PASS trên 20 câu |
|---|---:|---:|---:|---:|
| Đúng đắn | 14 | 5 | 1 | 70% |
| Đầy đủ | 13 | 6 | 1 | 65% |
| Trung thành với evidence | 18 | 2 | 0 | 90% |
| Citation đầy đủ về ngữ nghĩa | 13 | 6 | 1 | 65% |
| Rõ ràng | 18 | 2 | 0 | 90% |

- 11/20 câu đạt đồng thời bốn tiêu chí cốt lõi: đúng, đủ, trung thành và citation
  đầy đủ.
- Hợp đồng citation tự động đạt 20/20: có citation khi trả lời và mọi citation đều
  thuộc final evidence. Tuy nhiên citation đầy đủ về **ngữ nghĩa** chỉ đạt 13/20;
  điều này chứng minh kiểm tra schema không thay thế đánh giá nội dung.
- Hai câu trả lời tự khai báo `insufficient`: P015 và P016. Cả hai thực tế đều có
  GT, nên đây là thất bại tìm/chọn đủ evidence chứ không phải câu hỏi không trả lời
  được.
- Các trường hợp cần chú ý: P003 trộn ngữ cảnh thừa; P007 trả lời sai trọng tâm;
  P013 bỏ mờ điều kiện phải dự bài đánh giá; P014 thiếu danh sách chứng chỉ; P015,
  P016 và P019 thiếu một vế evidence; P018 kết luận đúng nhưng citation không chứng
  minh đủ cả hai tài liệu; P017 cần người xác minh phạm vi của hai mốc thời gian.

## Độ trễ, token và chi phí

### B0 cục bộ

| Thành phần | Trung bình | Trung vị | P95 | Lớn nhất |
|---|---:|---:|---:|---:|
| BM25 | 0,017 s | 0,011 s | 0,025 s | 0,107 s |
| Dense | 0,127 s | 0,057 s | 0,175 s | 1,629 s |
| Reranker | 1,023 s | 0,929 s | 1,184 s | 2,873 s |
| Tổng | 1,168 s | 0,993 s | 1,361 s | 4,610 s |

### S2G + GPT-5-nano

- 92 provider calls trong 20 artifact cuối.
- 137.455 input tokens; 1.280 cached input tokens; 66.988 output tokens, trong đó
  48.064 reasoning tokens.
- Chi phí API ước tính của 20 artifact cuối: **0,033610 USD**.
- Trung bình 23,91 giây/câu; trung vị 18,53 giây; P95 47,39 giây; lớn nhất 55,20
  giây.
- Trung bình 1,35 vòng retrieval/câu; tối đa 4 vòng.
- Stop reason: 17 `STOP_SUFFICIENT`, 2 `STOP_NO_NEW_EVIDENCE`, 1
  `STOP_MAX_TURNS`.

P004 ở lần đầu phát sinh `Duplicate evidence ID` tại sentence selector. Lần thất
bại này dùng thêm 2 provider calls, 2.430 input tokens, 1.781 output tokens và chi
phí ước tính 0,000834 USD. Tính cả lần thất bại, thực nghiệm dùng 94 calls và chi
phí API ước tính **0,034444 USD**. P004 hoàn thành khi chạy lại từ đầu; lỗi đầu tiên
được giữ trong audit, không được tính thành điểm retrieval bằng 0.

## Cơ sở cải tiến

Ưu tiên cải tiến được rút ra từ lỗi có quan sát, không phải tối ưu tham số trên pilot:

1. **Giữ đủ evidence cho câu nhiều vế.** Controller cần biểu diễn các yêu cầu con
   của câu hỏi và chỉ kết luận đủ khi mỗi yêu cầu có evidence độc lập. Không được
   dùng GT ở runtime.
2. **Giảm mất mát tại selector/citation.** Candidate pool đạt EGC@10 0,825 nhưng
   selected/citations chỉ còn 0,800. Cần trace rõ evidence nào bị loại và lý do,
   đồng thời bảo toàn evidence hỗ trợ các vế khác nhau.
3. **Thêm kiểm tra claim–citation.** Mỗi mệnh đề chính trong câu trả lời phải được
   gắn với evidence thực sự chứng minh mệnh đề đó; kiểm tra “citation thuộc final
   evidence” hiện chưa đủ.
4. **Cải thiện độ đầy đủ trước khi sinh.** P014–P016, P018–P019 cho thấy reasoner
   có thể trả lời trôi chảy từ một phần evidence nhưng bỏ vế còn lại.
5. **Làm bền giao diện sentence selector.** Evidence ID trùng nên được xử lý theo
   chính sách deterministic đã kiểm thử (ví dụ từ chối và retry có audit, hoặc
   canonical dedup nếu đặc tả cho phép), tránh làm hỏng toàn bộ request.
6. **Đánh giá trên tập giữ lại.** Sau khi sửa, cần dùng benchmark/holdout mới hoặc
   protocol pre-registered; không báo cáo cải thiện trên chính 20 câu đã dùng để
   chẩn đoán.

## Artifact

- `b0_rankings.json`, `b0_metrics.json`
- `s2g_rankings.json`, `s2g_retrieval_metrics.json`
- `s2g_resource_metrics.json`
- `answer_quality_review.csv` — các cột HUMAN vẫn để trống
- `answer_quality_agent_review.json`, `answer_quality_agent_summary.json`
- `s2g_runtime/runtime/P001` đến `P020` — trace và kết quả từng câu
- `s2g_runtime/runtime/P004/failure_attempt_1.json` — audit lỗi selector lần đầu
- `s2g_retry_p004/runtime/P004/` — trace đầy đủ của lần chạy lại thành công

Tất cả nằm dưới `outputs/evaluation/s2g_pilot_rerun_20260913/`.
