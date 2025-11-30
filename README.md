# HR Analytics: Dự đoán Giữ chân Nhân tài (Employee Retention Prediction)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Hardcoded-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

> **Mô tả ngắn:** Dự án xây dựng hệ thống dự đoán khả năng nghỉ việc của nhân sự trong ngành Data Science. Điểm đặc biệt của dự án là việc **tự cài đặt thuật toán Logistic Regression từ con số 0 (from scratch) chỉ sử dụng NumPy**, tích hợp các kỹ thuật nâng cao như Regularization, Class Weighting và Threshold Tuning để giải quyết bài toán mất cân bằng dữ liệu.

---

## Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)


---

## Giới thiệu

### Mô tả bài toán
Trong nền kinh tế tri thức, "chảy máu chất xám" là cơn ác mộng của mọi doanh nghiệp. Chi phí để tuyển dụng và đào tạo lại một nhân sự Data Scientist là rất lớn. Bài toán đặt ra là: *Làm thế nào để nhận diện sớm những nhân viên có ý định nghỉ việc để HR kịp thời có chính sách giữ chân?*

### Mục tiêu cụ thể
1.  Phân tích các yếu tố ảnh hưởng đến quyết định nghỉ việc (EDA).
2.  Xây dựng mô hình phân loại nhị phân (Binary Classification) để dự báo:
    * `0`: Ổn định (Ở lại).
    * `1`: Rủi ro (Muốn nghỉ việc).
3.  Tối ưu hóa chỉ số **Recall** (để không bỏ sót nhân tài muốn đi) trong bối cảnh dữ liệu bị mất cân bằng nghiêm trọng.

---

##  Dataset

* **Nguồn dữ liệu:** [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)
* **Kích thước:** ~19,158 mẫu (bản ghi).
* **Tổng số đặc trưng gốc:** 14 cột
* **Đặc điểm quan trọng:**
    * **Imbalanced Data:** Chỉ có ~25% nhân sự muốn nghỉ việc, 75% ở lại.
    * **Features:** Bao gồm cả định lượng (Training hours, City index) và định tính (Gender, Education, Experience).
    * **Missing Values:** Một số cột như `company_type`, `gender` thiếu dữ liệu lên đến 30%.

| STT | Feature                  | Mô tả                                                                                   |
|-----|--------------------------|------------------------------------------------------------------------------------------|
| 1   | `enrollee_id`            | Mã định danh duy nhất của ứng viên                                                      |
| 2   | `city`                   | Mã thành phố nơi ứng viên đang sinh sống                                                |
| 3   | `city_development_index` | Chỉ số phát triển kinh tế - xã hội của thành phố                       |
| 4   | `gender`                 | Giới tính của ứng viên (M / F / Other)                                          |
| 5   | `relevent_experience`    | Ứng viên có kinh nghiệm liên quan đến lĩnh vực Data Science hay không                  |
| 6   | `enrolled_university`    | Loại hình đào tạo đại học/cao học hiện tại  |
| 7   | `education_level`        | Trình độ học vấn cao nhất                                       |
| 8   | `major_discipline`       | Ngành học chính ở bậc đại học (STEM, Humanities, Business, Arts, v.v.)                 |
| 9   | `experience`             | Tổng số năm kinh nghiệm làm việc (từ <1 năm đến >20 năm)                                |
| 10  | `company_size`           | Quy mô nhân sự của công ty hiện tại                 |
| 11  | `company_type`           | Loại hình công ty hiện tại  |
| 12  | `last_new_job`           | Khoảng cách (năm) từ lần chuyển việc gần nhất               |
| 13  | `training_hours`         | Tổng số giờ đào tạo mà ứng viên đã hoàn thành trong nền tảng hiện tại                   |
| 14  | `target`                 | Nhãn mục tiêu: 0 = Không tìm việc mới (ở lại), 1 = Đang tìm việc mới (có ý định nghỉ việc) |

---

## Method

### 1. Quy trình xử lý dữ liệu (Preprocessing Pipeline)
* **Cleaning:** Điền khuyết (Imputation) chiến lược: dùng Mode cho biến ngẫu nhiên và tạo nhóm 'Unknown' cho biến thiếu có hệ thống.
* **Feature Engineering:**
    * Gom nhóm `city` (Top 10 + Others).
    * Tạo đặc trưng tương tác: `Brain Drain` (Học vấn cao + Vùng kém phát triển).
* **Encoding:**
    * **Ordinal Encoding:** Áp dụng cho biến có thứ tự (`experience`, `education`, `company_size`,`last_new_job`,`enrolled_university`) để giữ nguyên tính chất lớn bé.
    * **Label/One-Hot Encoding:** Cho các biến định danh.
* **Scaling:** StandardScaler để đưa dữ liệu về phân phối chuẩn ($\mu=0, \sigma=1$).

#### 1. Hàm kích hoạt – Sigmoid
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Trong đó:
- $z = \mathbf{w}^T \mathbf{x} + b$: Giá trị đầu vào tuyến tính (logit)  
  - $\mathbf{w}$: Vector trọng số của mô hình (kích thước = số feature)  
  - $\mathbf{x}$: Vector đặc trưng của một mẫu dữ liệu  
  - $b$: Bias (hệ số tự do)  

#### 2. Hàm mất mát – Weighted Binary Cross-Entropy + L2 Regularization
$$
J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \alpha_i \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

**Ý nghĩa từng thành phần:**
| Ký hiệu           | Mô tả                                                                                 |
|-------------------|---------------------------------------------------------------------------------------|
| $m$               | Số lượng mẫu trong tập huấn luyện                                                     |
|$n$|Số lượng đặc trưng (features) trong dữ liệu|
| $y^{(i)}$         | Nhãn thực tế của mẫu thứ $i$ (0 = ở lại, 1 = nghỉ việc)                               |
| $\hat{y}^{(i)}$   | Xác suất dự đoán lớp 1: $\hat{y}^{(i)} = \sigma(\mathbf{w}^T \mathbf{x}^{(i)} + b)$  |
| $\alpha_i$        | **Trọng số mẫu** (sample_weight) – tự động tính theo `class_weight='balanced'` hoặc truyền vào |
| $\lambda$         | Hệ số phạt L2 (Ridge) – kiểm soát overfitting                                        |


→ Dùng `np.clip(prob, 1e-15, 1-1e-15)` để tránh `log(0)`

#### 3. Cập nhật tham số – Gradient Descent (hoàn toàn vector hóa)

Quá trình huấn luyện sử dụng Gradient Descent để tối thiểu hóa hàm mất mát:

$$
\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \frac{\partial J}{\partial \mathbf{w}}
\qquad
b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}
$$

trong đó $\alpha$ là **learning rate** (tốc độ học).
### Gradient thực tế (weighted loss + L2 regularization)

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^T \Big( (\hat{\mathbf{y}} - \mathbf{y}) \boldsymbol{\alpha} \Big) + \frac{\lambda}{m} \mathbf{w}
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \alpha_i \, (\hat{y}^{(i)} - y^{(i)})
$$

### Giải thích tổng quát các ký hiệu

| Ký hiệu            | Ý nghĩa tổng quát                                                                 |
|--------------------|------------------------------------------------------------------------------------|
| $\mathbf{X}$       | Ma trận dữ liệu đầu vào, shape $(m \times n)$                                     |
| $\mathbf{w}$       | Vector trọng số mô hình, shape $(n \times 1)$                                     |
| $b$                | Bias (hệ số chặn)                                                                 |
| $\mathbf{y}$       | Vector nhãn thực tế (0/1), shape $(m \times 1)$                                   |
| $\hat{\mathbf{y}}$ | Vector xác suất dự đoán lớp 1 = $\sigma(\mathbf{X}\mathbf{w} + b)$                |
| $m$                | Số lượng mẫu trong batch/tập huấn luyện                                           |
| $n$                | Số lượng đặc trưng                                                                |
| $\boldsymbol{\alpha}$ | Vector trọng số mẫu (sample_weight), shape $(m \times 1)$                      |
| $\alpha_i$         | Trọng số của mẫu thứ $i$ (tự động tính từ `class_weight='balanced'` nếu cần)     |
| $\lambda$          | Hệ số phạt L2 (Ridge regularization)                                              |


Tất cả được **vector hóa 100% bằng NumPy**, mỗi vòng lặp chỉ thực hiện đúng **2 phép toán ma trận** để cập nhật toàn bộ tham số — nhanh và chính xác tương đương thư viện chuẩn.
#### 4. Tính năng nâng cao đã tự triển khai

| Tính năng                        | Mô tả                                                                 |
|----------------------------------|-----------------------------------------------------------------------|
| `sample_weight` / `class_weight='balanced'` | Tự động tính trọng số ngược tần suất lớp để xử lý mất cân bằng       |
| L2 Regularization                | Kiểm soát overfitting hiệu quả                                       |
| Threshold Tuning                 | Tự tìm threshold tối ưu theo F1-score trên tập validation            |
| Learning Curve                   | Vẽ đồ thị cost giảm dần qua các iteration                            |
| Cross-Validation       | Đánh giá F1-score trung bình  |
## Installation & Setup

1.  **Clone dự án:**
    ```bash
    git clone https://github.com/Phucvt123/Project
    cd Project
    ```

2.  **Tạo môi trường ảo (Khuyên dùng):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Thư viện chính: numpy, matplotlib, seaborn, scikit-learn)*

---

## Usage

Dự án được chia thành các Notebook theo quy trình chuẩn:

1.  **Khám phá dữ liệu (EDA):**
    * Chạy file `notebooks/01_data_exploration.ipynb`.
    * Xem phân tích Heatmap, Cramér's V correlation và các insight về Brain Drain.

2.  **Tiền xử lý (Preprocessing):**
    * Chạy file `notebooks/02_preprocessing.ipynb`.
    * File này sẽ tạo ra `aug_train.csv` và `aug.test.csv` trong thư mục `data/processed/`.

3.  **Huấn luyện & Đánh giá (Modeling):**
    * Chạy file `notebooks/03_modeling.ipynb`.
    * So sánh kết quả giữa Logistic Regression (Custom NumPy), Logistic Regression(thư viện có sẵn) và Random Forest(thư viện có sẵn).

---

## Kết quả đạt được (trên tập Validation – Threshold đã được tối ưu theo F1-score)

| Mô hình                          | Accuracy | Precision | Recall  | F1-Score | AUC     | Ghi chú                                      |
|----------------------------------|----------|-----------|---------|----------|---------|----------------------------------------------|
| **Logistic Custom 1**            | 0.7554   | 0.5076    | 0.6262  | 0.5607   | 0.7576  | Giống hệt 100% với sklearn                   |
| **Logistic Custom 2**            | 0.7520   | 0.5019    | 0.7068 | 0.5870 | 0.7710  | Recall cao nhất toàn dự án            |
| **Logistic Regression (sklearn)**    | 0.7554   | 0.5076    | 0.6262  | 0.5607   | 0.7576  | Xác nhận thuật toán tự viết chính xác |
| **Random Forest**        | 0.8003 | 0.5906 | 0.6482 | 0.6181 | 0.7880 | Mô hình tốt nhất tổng thể |

### Phân tích & Kết luận chính

- **Random Forest** vượt trội hoàn toàn về **F1-score (0.6181)** và **AUC (0.7880)**.
- **Logistic Regression tự viết (Custom 1)** đạt **giống hệt** so với `sklearn.LogisticRegression` → chứng minh thuật toán từ đầu hoàn toàn chính xác.
- **Logistic Custom 2** đạt **Recall cao nhất: 70.68%** → lựa chọn lý tưởng trong trường hợp doanh nghiệp ưu tiên **“không bỏ sót bất kỳ nhân viên nào có nguy cơ nghỉ việc”**.
## Project Structure
```
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   └── processed/            # Dữ liệu sau khi làm sạch và mã hóa
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Phân tích EDA
│   ├── 02_preprocessing.ipynb     # Xử lý dữ liệu
│   └── 03_modeling.ipynb          # Huấn luyện mô hình
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Các hàm xử lý, encoding, scaling
│   ├── models.py             # Class LogisticRegressionCustom (NumPy)
│   └── visualization.py      # Các hàm vẽ biểu đồ
├── README.md                 # Tài liệu dự án
└── requirements.txt          # Danh sách thư viện
```
## Challenges & Solutions

Trong quá trình **tự xây dựng hoàn toàn Logistic Regression chỉ bằng NumPy**, mình đã gặp và khắc phục thành công các thử thách kỹ thuật sau:

| Thách thức                                | Mô tả vấn đề                                                                 | Giải pháp đã áp dụng                                                                 |
|-------------------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Tràn số & mất ổn định số học**          | Hàm `sigmoid` và `log` dễ trả về `inf`/`-inf`/`nan` khi $z$ quá lớn hoặc quá nhỏ | Dùng `np.clip(z, -250, 250)` và `np.clip(prob, 1e-15, 1-1e-15)` trước khi tính log   |
| **Tốc độ cực chậm khi dùng vòng lặp Python** | Tính gradient bằng for-loop trên hàng chục nghìn mẫu → mất hàng phút mỗi epoch | **Vector hóa 100%**: dùng `X @ w`, `X.T @ error`, broadcasting → giảm từ phút xuống mili-giây |
| **Mất cân bằng dữ liệu nghiêm trọng (25%/75%)** | Mô hình luôn đoán lớp đa số (0) → Recall gần 0                                | Tự cài `sample_weight` + `class_weight='balanced'` trực tiếp vào cost & gradient      |
| **Threshold mặc định 0.5 không tối ưu**   | F1-score thấp dù accuracy cao                                                 | Tự viết hàm `find_best_threshold()` duyệt 300 ngưỡng → chọn threshold tối ưu F1       |


## Future Improvements

- Thử nghiệm **Deep Learning** (MLP, TabNet, Transformer) bằng PyTorch/TensorFlow  
- **Deploy thực tế**:  
  – FastAPI + Docker → API dự đoán nhanh  
  – Streamlit/Dashboard → giao diện cho phòng Nhân sự nhập thông tin và xem kết quả ngay  

- Tích hợp vào hệ thống HR hiện tại (Workday, SAP SuccessFactors…)

## Contributors

**Tác giả & Chủ dự án**  
- **Họ và tên**: Vũ Trần Phúc
- **MSSV**: 23120333
- **Lớp**: 23_21
- **Vai trò**:  
  → Tự triển khai thuật toán Logistic Regression từ con số 0 chỉ dùng NumPy  
  → Huấn luyện, đánh giá  mô hình


## Contact

- Email: 23120333@student.hcmus.edu.vn

## License

Dự án được phát hành dưới **MIT License** – bạn hoàn toàn được tự do sử dụng, chỉnh sửa và thương mại hóa.
