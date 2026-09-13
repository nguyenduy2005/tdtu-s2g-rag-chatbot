# Quy trình đánh giá S2G-RAG

## Phạm vi và trạng thái

Quy trình này đo lường sản phẩm full-PDF hiện tại theo năm khía cạnh:

1. recall của retrieval;
2. thứ hạng nghịch đảo trung bình (MRR);
3. độ bao phủ nhóm evidence (EGC);
4. chất lượng câu trả lời được con người đánh giá;
5. độ trễ end-to-end và chi phí OpenAI API.

`data/evaluation/s2g_pilot_v1/queries.json` là bộ pilot gồm 20 câu hỏi và 27 nhóm
evidence dựa trên Canonical Corpus V1.1. Trạng thái hiện tại là
`DRAFT_AGENT_VERIFIED_PENDING_HUMAN`. Các điểm số thô có thể dùng để chẩn đoán kỹ
thuật nhưng không được mô tả là kết quả của một nghiên cứu annotation độc lập bởi con
người. Báo cáo khóa luận chính thức yêu cầu hoàn tất `ground_truth_review.csv` và chỉ
đổi trạng thái từng câu thành `human_verified` sau khi kiểm tra.

## Ground-truth firewall

File benchmark chứa nhãn evidence. File runtime chỉ chứa đúng hai trường `id` và
`query`; loader từ chối mọi trường bổ sung. Vì vậy, quá trình chạy retrieval và S2G
không thể nhìn thấy ground truth. Offline evaluator chỉ đọc nhãn evidence sau khi
bảng xếp hạng/kết quả đã được hoàn tất.

CLI mặc định yêu cầu trạng thái `human_verified`. Cờ `--allow-pending-human` chỉ là
lựa chọn tường minh để chạy phép đo kỹ thuật tạm thời.

## Chỉ số retrieval

- `Recall@k`: số chunk ID liên quan duy nhất xuất hiện trong k kết quả đầu, chia cho
  toàn bộ chunk ID liên quan duy nhất của câu hỏi.
- `MRR@10`: nghịch đảo thứ hạng của chunk liên quan đầu tiên trong 10 kết quả đầu;
  bằng 0 nếu không tìm thấy.
- `Evidence-group Coverage@k`: số nhóm evidence bắt buộc được bao phủ trong k kết
  quả đầu, chia cho tổng số nhóm bắt buộc. Một nhóm được bao phủ nếu xuất hiện ít nhất
  một chunk ID được chấp nhận của nhóm.
- `Complete-evidence@k`: chỉ đúng khi toàn bộ nhóm bắt buộc được bao phủ.

Các chỉ số được báo cáo theo từng câu hỏi, trung bình macro và theo query type. Với
S2G, báo cáo tách riêng iterative candidate pool, các chunk do reranker trình bày,
parent chunk được sentence selector chọn và citation của câu trả lời. Các giai đoạn
này không thể dùng thay thế cho nhau.

## Chất lượng câu trả lời

Kiểm tra tự động chỉ bao phủ các hợp đồng có thể cưỡng chế: citation ID phải thuộc
final evidence và câu trả lời có nội dung phải kèm citation. Chất lượng ngữ nghĩa được
kiểm tra trong `answer_quality_review.csv`. Người đánh giá ghi TRUE/FALSE cho tính
đúng đắn, đầy đủ, trung thành với evidence, mức đầy đủ của citation và độ rõ ràng,
kèm ghi chú. Toàn bộ trường HUMAN để trống khi file được tạo. Báo cáo phải luôn có cả
tử số/mẫu số và tỷ lệ phần trăm; không được âm thầm loại câu trả lời insufficient.

## Độ trễ và chi phí

Trace retrieval ghi độ trễ của BM25, dense, fusion, reranker và tổng thời gian. Trace
S2G ghi độ trễ từng lần retrieval, độ trễ gọi provider và độ trễ end-to-end. Báo cáo
sử dụng trung bình, trung vị, P95 và giá trị lớn nhất theo câu hỏi.

Chi phí token dùng usage do Responses API ghi nhận gồm `input_tokens`, cached input
token và `output_tokens`. Cấu hình lưu ngày quan sát giá và nguồn giá. Ước tính này
không bao gồm chi phí máy cục bộ, điện năng và mạng; do đó phải được gọi là chi phí
API ước tính, không phải tổng chi phí hệ thống.

## Các lệnh tái lập

Sử dụng môi trường của dự án:

```bash
conda run -n ml-env python -m src.evaluation.cli validate
conda run -n ml-env python -m src.evaluation.cli export-runtime \
  --output data/evaluation/s2g_pilot_v1/runtime_queries.json
conda run -n ml-env python -m src.evaluation.cli run-b0 \
  --runtime-queries data/evaluation/s2g_pilot_v1/runtime_queries.json \
  --output outputs/evaluation/s2g_pilot_v1/b0_rankings.json
conda run -n ml-env python -m src.evaluation.cli evaluate-b0 \
  --rankings outputs/evaluation/s2g_pilot_v1/b0_rankings.json \
  --output outputs/evaluation/s2g_pilot_v1/b0_metrics.json
conda run -n ml-env python -m src.evaluation.cli run-s2g \
  --runtime-queries data/evaluation/s2g_pilot_v1/runtime_queries.json \
  --output-root outputs/evaluation/s2g_pilot_v1/s2g_runtime
conda run -n ml-env python -m src.evaluation.cli evaluate-s2g \
  --runtime-root outputs/evaluation/s2g_pilot_v1/s2g_runtime \
  --output-root outputs/evaluation/s2g_pilot_v1
```

Cho đến khi hoàn tất kiểm chứng bởi con người, chỉ thêm `--allow-pending-human` vào
các lệnh validate, export và offline evaluation. Lệnh thực thi chỉ đọc file runtime
đã loại nhãn và không bao giờ tải benchmark.

## Giới hạn diễn giải

Hai mươi câu hỏi được phân tầng chỉ là pilot, không phải ước lượng toàn diện cho mọi
câu hỏi có thể có của sinh viên. Việc soạn câu hỏi, kiểm tra ground truth, hành vi mô
hình, nhiễu OCR và sự mơ hồ về vòng đời tài liệu là các nguy cơ độc lập đối với tính
hợp lệ. Không được tinh chỉnh prompt, cấu hình retrieval hoặc câu hỏi sau khi xem kết
quả từng câu trong pilot rồi báo cáo trên chính tập đó như một kết quả không thiên lệch.
