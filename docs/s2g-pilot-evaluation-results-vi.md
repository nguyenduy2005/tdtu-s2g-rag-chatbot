# Kết quả đánh giá thử nghiệm S2G-RAG

## Trạng thái

**Thực thi: PASS (hoàn thành 20/20 câu hỏi).** Đây là thử nghiệm kỹ thuật, chưa phải
bộ benchmark khóa luận đã đóng băng. Benchmark sử dụng bước kiểm chứng nguồn có hỗ
trợ của agent và hiện có **0/20 câu hỏi được con người kiểm chứng độc lập**. Vì vậy,
validator chính thức nghiêm ngặt chủ động trả về lỗi theo thiết kế. Các kết quả dưới
đây phải được ghi rõ là *tạm thời* cho đến khi hoàn tất `ground_truth_review.csv`.

- Corpus: Canonical Corpus V1.1, gồm 27 PDF và 3.482 chunk được lập chỉ mục.
- SHA-256 corpus: `fd2358332668de3dca8c099ff6a3dd0cf1d23b47c378f7c92835df4a08c34c02`.
- Bộ thử nghiệm: 20 câu hỏi, 27 nhóm evidence bắt buộc, 23 GT chunk duy nhất.
- Phân bố: 5 direct lookup, 5 paraphrase, 3 terminology mismatch,
  4 multi-evidence và 3 cross-document.
- GT firewall lúc chạy: PASS. Đầu vào runtime chỉ chứa `id` và `query`.
- Mô hình yêu cầu: `gpt-5-nano`; cả 89 phản hồi đều được phân giải thành
  `gpt-5-nano-2025-08-07`; tất cả response ID đều tồn tại và duy nhất.
- Lỗi API/hạ tầng: 0.

## Định nghĩa chỉ số

- Recall@k là tỷ lệ số GT chunk duy nhất được tìm thấy trong top-k, lấy trung bình macro.
- MRR@10 sử dụng nghịch đảo thứ hạng của GT chunk đầu tiên.
- Evidence-group Coverage (EGC)@k là tỷ lệ nhóm evidence bắt buộc có ít nhất một
  chunk hợp lệ xuất hiện trong top-k.
- Complete Evidence chỉ đạt khi toàn bộ nhóm evidence của câu hỏi đều được bao phủ.

Mỗi nhóm evidence trong pilot hiện chỉ có một chunk được chấp nhận, nên Recall@k và
EGC@k có cùng giá trị số. Hai chỉ số vẫn được cài đặt riêng vì trong tương lai một
nhóm có thể có nhiều chunk thay thế hợp lệ.

## Kết quả retrieval B0

| Giai đoạn | R@1 | R@3 | R@5 | R@10 | MRR@10 | EGC@10 | Đủ evidence @10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.525 | 0.800 | 0.850 | 0.875 | 0.746 | 0.875 | 16/20 |
| Dense | 0.500 | 0.625 | 0.775 | 0.850 | 0.672 | 0.850 | 16/20 |
| Hybrid + RRF | 0.525 | 0.825 | 0.825 | 0.875 | 0.742 | 0.875 | 16/20 |
| Hybrid + RRF + reranker | **0.575** | **0.825** | **0.825** | **0.825** | **0.767** | **0.825** | **15/20** |

Reranker cải thiện kết quả ở hạng 1 và MRR so với hybrid thô, nhưng đẩy evidence đầy
đủ của một câu hỏi ra khỏi top-10 (từ 16 còn 15 câu). Đây là đánh đổi được đo trực
tiếp, không phải căn cứ để tinh chỉnh trên chính bộ pilot này.

Ở giai đoạn B0 cuối, R@10 của direct lookup là 1.000, terminology mismatch là 1.000,
paraphrase là 0.800, cross-document là 0.667 và multi-evidence là 0.625. Chỉ 2/4 câu
multi-evidence và 1/3 câu cross-document có đầy đủ evidence trong top-10.

## Kết quả retrieval/evidence của S2G

Candidate pool và evidence cuối được báo cáo riêng. Iterative candidate pool chứa
sáu chunk có thứ hạng reranker cao nhất ở mỗi vòng retrieval đã thực thi. Selected
evidence chứa các parent chunk được sentence selector chấp nhận; answer citations là
tập con được answer reasoner sử dụng.

| Giai đoạn | R@1 | R@3 | R@5 | R@10 | MRR@10 | EGC@10 | Đủ evidence @10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Iterative candidate pool | 0.625 | 0.750 | 0.800 | 0.800 | 0.777 | 0.800 | 14/20 |
| S2G selected evidence | 0.625 | 0.700 | 0.750 | 0.750 | 0.785 | 0.750 | 13/20 |
| Answer citations | 0.600 | 0.750 | 0.750 | 0.750 | 0.767 | 0.750 | 13/20 |

S2G không vượt B0 về retrieval top-10 trong pilot này. So với B0 hybrid+reranker,
R@10 của candidate S2G giảm từ 0.825 xuống 0.800, còn R@10 của selected/cited
evidence giảm xuống 0.750. MRR@10 của selected evidence cao hơn nhẹ (0.785 so với
0.767), cho thấy evidence tìm được có độ chính xác sớm tốt hơn nhưng độ bao phủ tổng
thể yếu hơn.

Vấn đề chính là độ bao phủ multi-evidence. Ở giai đoạn selected evidence, R/EGC@10
chỉ đạt 0.250 và số câu đủ evidence là 0/4. Direct lookup và terminology mismatch đều
đạt 1.000; paraphrase đạt 0.800; cross-document đạt 0.667.

### Phân tích luồng evidence

- P007 và P015: GT không xuất hiện trong top-10 B0 lẫn candidate pool của S2G.
- P014 và P018-P019: chỉ một trong hai nhóm evidence bắt buộc được tìm thấy/trích dẫn.
- P016: cả hai GT chunk đều có trong iterative candidate pool nhưng không chunk nào
  được đưa vào selected evidence. Đây là mất mát tại selector/controller, không phải
  lỗi của retrieval giai đoạn đầu.
- P017: top-10 B0 chứa cả hai GT chunk, còn candidate pool giới hạn sáu chunk mỗi vòng
  của S2G chỉ chứa một. Việc kiểm tra cũng phát hiện khả năng ground truth bị mơ hồ
  giữa một quy định tổng quát và một quy định cụ thể.

Không tham số retrieval hoặc prompt nào được thay đổi sau khi quan sát các trường hợp này.

## Đánh giá chất lượng câu trả lời có hỗ trợ của agent

Đánh giá này hữu ích cho việc chẩn đoán nhưng **không phải đánh giá độc lập của con
người**. Tỷ lệ PASS dùng toàn bộ 20 câu hỏi làm mẫu số; REVIEW không được tính là PASS.

| Tiêu chí | PASS | FAIL | REVIEW | PASS / 20 |
|---|---:|---:|---:|---:|
| Tính đúng đắn | 14 | 5 | 1 | 70% |
| Tính đầy đủ | 13 | 6 | 1 | 65% |
| Trung thành với evidence | 15 | 4 | 1 | 75% |
| Mức đầy đủ của trích dẫn | 12 | 7 | 1 | 60% |
| Độ rõ ràng | 18 | 2 | 0 | 90% |

Mười trong 20 câu trả lời đạt đồng thời bốn tiêu chí cốt lõi: đúng đắn, đầy đủ,
trung thành với evidence và trích dẫn đầy đủ. Vấn đề nổi bật là không trả lời đủ các
vế của câu multi-evidence, chỉ trích dẫn một tài liệu/nhóm bắt buộc, hoặc trả lời từ
một quy định lân cận có vẻ hợp lý nhưng không đúng evidence cần thiết. P013 diễn giải
sai rằng chỉ cần có chứng chỉ MOS là được bỏ qua môn học; nguồn yêu cầu phải thi đánh
giá và đạt ngưỡng. P015 sai đáng kể, P016 trả lời thiếu căn cứ và P019 bỏ sót phần đối
tượng đại học. P017 giữ trạng thái REVIEW vì hai quy định nguồn có mức độ cụ thể khác
nhau về thời điểm nộp hồ sơ.

Phiếu đánh giá con người chưa bị chỉnh sửa là file `answer_quality_review.csv`
trong thư mục kết quả pilot.

## Độ trễ, token và chi phí API

### Độ trễ retrieval cục bộ của B0 (giây/câu hỏi)

| Thành phần | Trung bình | Trung vị | P95 | Lớn nhất |
|---|---:|---:|---:|---:|
| BM25 | 0.011 | 0.010 | 0.020 | 0.022 |
| Dense | 0.088 | 0.062 | 0.131 | 0.766 |
| Reranker | 0.999 | 0.956 | 1.215 | 1.913 |
| Tổng | **1.099** | **1.017** | **1.368** | **2.694** |

Chi phí API của B0 bằng 0 vì các mô hình retrieval chạy cục bộ. Con số này không bao
gồm chi phí phần cứng, điện năng hoặc công sức kỹ thuật.

### Tài nguyên S2G

- Độ trễ end-to-end: trung bình 22.978 giây, trung vị 21.666 giây, P95 41.059 giây,
  lớn nhất 45.716 giây.
- Độ trễ gọi provider: trung bình 20.895 giây/câu.
- Độ trễ retrieval qua các vòng lặp: trung bình 2.046 giây/câu.
- Số vòng retrieval: trung bình 1,30, trung vị 1, lớn nhất 3.
- Lý do dừng: 17 `STOP_SUFFICIENT`, 3 `STOP_NO_NEW_EVIDENCE`.
- Số lần gọi provider: tổng 89, trung bình 4,45/câu, gồm 43 judge, 26 sentence
  selector và 20 answer reasoner.
- Token: 143.847 input, 65.214 output, trong đó có 46.720 reasoning token; provider
  không báo cáo cached input token.
- Chi phí OpenAI API ước tính: **0,03327795 USD cho toàn bộ**, tương đương
  **0,00166390 USD/câu**.

Ước tính dùng lượng token được ghi nhận và giá GPT-5 nano tại ngày 2026-09-13:
0,05 USD/triệu input token, 0,005 USD/triệu cached input token và 0,40 USD/triệu
output token. Chi phí này không bao gồm tính toán cục bộ và mạng.

## Kết luận về tính hợp lệ và checkpoint tiếp theo

Phần mềm và quá trình thực thi đều đạt; bằng chứng cho thấy S2G đang hoạt động tốt
với câu hỏi direct/terminology nhưng chưa đáng tin cậy cho câu hỏi multi-evidence.
Kết quả phù hợp cho chẩn đoán kỹ thuật và báo cáo tiến độ, nhưng **chưa phù hợp để
dùng làm kết luận cuối cùng của khóa luận** vì hai lý do:

1. benchmark chưa được con người kiểm chứng ground truth độc lập;
2. chất lượng câu trả lời được đánh giá với hỗ trợ của agent và P017 cho thấy ít nhất
   một điểm mơ hồ trong annotation.

Bước tiếp theo là kiểm tra toàn bộ dòng trong `ground_truth_review.csv`, xử lý P017
mà không nhìn vào bảng xếp hạng retrieval, rồi đóng băng benchmark. Sau đó cần chạy
một thực nghiệm chính thức mới, không tinh chỉnh, và đánh giá độc lập chất lượng câu
trả lời. Không được thay đổi tham số retrieval/controller dựa trên pilot này rồi dùng
lại chính các câu hỏi đó như một tập kiểm tra không thiên lệch.
