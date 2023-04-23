# Kết quả đánh giá pipeline trích xuất dữ liệu text từ sổ hồng
Mục tiêu: Xem xét ảnh hưởng của việc scan tới kết quả đầu ra

Tập ảnh đánh giá:
- Crawl từ google và chọn lọc những ảnh có đầy đủ thông tin và có thể đọc được bằng mắt thường
- Chất lượng ảnh được đánh giá qua 3 khía cạnh:
    - Mờ hay rõ nét
    - Chữ xiên hay thẳng
    - Giấy cong hay phẳng

Ảnh mẫu:

![Ảnh mẫu](../assets/data/000293.jpg)

## 1. Lần 1

Pipeline:
1. Scan/Không scan
2. Tách trang
3. Text detection
4. Merge bboxes v1 (Rectangle box)
5. Group lines
6. OCR
7. Correction (ChatGPT)
8. Information extractions (ChatGPT)

Text detection:

![Text detection](../assets/result/det_dbpp/v2/raw/000293/det_raw.jpg)

Merge box v1:

![Merge box v1](../assets/result/det_dbpp/v1/raw/000293/det_merged.jpg)

Text box:

![Text box v1](../assets/result/det_dbpp/v1/raw/000293/subimgs/001.jpg)

![Text box v1](../assets/result/det_dbpp/v1/raw/000293/subimgs/003.jpg)
    
Kết quả đánh giá:

Chất lượng ảnh | Ảnh raw | Ảnh scan
:------ | :---: | :---: 
Tổng quát | 36.7% | 63.3%
Ảnh mờ | 33.3% | 33.3%
Ảnh cong | 0% | 83.3%
Ảnh xiên | 33.3% | 72.2%
Ảnh nét | 50% | 83.3%
Ảnh phẳng | 54.2% | 58.3%
Ảnh thẳng | 58.3% | 50%
Ảnh hoàn hảo | 83.3% | 83.3%

Nhận xét:
- Việc scan giúp giảm tác động của môi trường tới quá trình xử lý của pipeline
- Việc scan ảnh giúp cải thiện ảnh cong hoặc xiên, trong khi không có tác động gì trong trường hợp ảnh mờ, thiếu sáng
- Độ chính xác của pipeline khi ảnh phẳng và thẳng là xấp xỉ trong 2 trường hợp scan và không scan
- Trong trường hợp ảnh hoàn hảo (rõ nét, thẳng, phẳng) thì việc scan không mang lại lợi ích gì

**Kết luận:** Vậy, nếu có thể định hướng để người dùng chụp ảnh đẹp, rõ nét hoặc có thể tạo tác động tương tự như việc scan trong pipeline thì không cần thực hiện scan ảnh

## Lần 2
Pipeline:
1. Scan/Không scan
2. Tách trang
3. Text detection
4. **Vertical padding**
5. **Merge bboxes v2**
6. **Text alignment**
7. Group lines
8. OCR
9. Correction (ChatGPT)
10. Information extractions (ChatGPT)

Text detection:

![Text detection](../assets/result/det_dbpp/v2/raw/000293/det_raw.jpg)

Merge box v2:

![Merge box v1](../assets/result/det_dbpp/v2/raw/000293/det_merged.jpg)

Text box:

![Text box v1](../assets/result/det_dbpp/v2/raw/000293/subimgs/000.jpg)

![Text box v1](../assets/result/det_dbpp/v2/raw/000293/subimgs/002.jpg)

Kết quả đánh giá:

Chất lượng ảnh | Ảnh raw | Ảnh scan
:------ | :---: | :---: 
Tổng quát | 66.7% | 56.7%
Ảnh mờ | 50% | 25%
Ảnh cong | 50% | 83.3%
Ảnh xiên | 66.7% | 55.6%
Ảnh nét | 77.8% | 77.8%
Ảnh phẳng | 70.8% | 50%
Ảnh thẳng | 66.7% | 58.3%
Ảnh hoàn hảo | 100% | 83.3%

Nhận xét:
- Với pipeline 2 thì ảnh raw cho kết quả cao hơn ảnh scan
- Việc scan làm ảnh tệ hơn trong trường hợp ảnh mờ
- Trong nhiều trường hợp, ảnh scan kém hiệu quả hơn
- Trong trường hợp ảnh hoàn hảo, ảnh raw cho kết quả tuyệt đối

## 3. Tổng kết

Chất lượng ảnh | Ảnh raw (1) | Ảnh scan (1) | Ảnh raw (2) | Ảnh scan (2)
:------ | :---: | :---: | :---: | :---: 
Tổng quát | 36.7% | 63.3% | **66.7%** | 56.7%
Ảnh mờ | 33.3% | 33.3% | **50%** | 25%
Ảnh cong | 0% | **83.3%** | 50% | **83.3%**
Ảnh xiên | 33.3% | **72.2%** | 66.7% | 55.6%
Ảnh nét | 50% | **83.3%** | 77.8% | 77.8%
Ảnh phẳng | 54.2% | 58.3% | **70.8%** | 50%
Ảnh thẳng | 58.3% | 50% | **66.7%** | 58.3%
Ảnh hoàn hảo | 83.3% | 83.3% | **100%** | 83.3%

**Kết luận:**
- Ảnh raw (2) cho kết quả tốt nhất về tổng quan
- Trong trường hợp ảnh kém chất lượng, ảnh scan (1) vẫn cho kết quả nhỉnh hơn
- Trong trường hợp ảnh chất lượng tốt, ảnh raw (2) cho kết quả nhỉnh hơn