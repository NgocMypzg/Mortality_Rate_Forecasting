# Hệ thống Dự báo Tỷ lệ Tử vong Toàn cầu

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Công nghệ sử dụng](#2-công-nghệ-sử-dụng)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Chức năng chính](#4-chức-năng-chính)
5. [Quy trình hoạt động](#5-quy-trình-hoạt-động)
6. [Hướng dẫn cài đặt](#6-hướng-dẫn-cài-đặt)
7. [Ghi chú](#7-ghi-chú)
8. [Đóng góp](#8-đóng-góp)
9. [Liên hệ](#9-liên-hệ)

---

## 1. Giới thiệu

### Mục tiêu của hệ thống

Hệ thống Dự báo Tỷ lệ Tử vong Toàn cầu (Mortality Rate Forecasting System) là một ứng dụng web giúp người dùng:
- Visualize dữ liệu tỷ lệ tử vong do thảm họa theo quốc gia và khu vực
- Tra cứu dữ liệu lịch sử từ năm 2000 đến 2024
- Dự báo tỷ lệ tử vong trong các năm tới (2025-2030)
- Xuất dữ liệu lịch sử và dự báo dưới dạng CSV

### Bài toán mà ứng dụng giải quyết

Tỷ lệ tử vong do thảm họa là một chỉ số quan trọng trong công tác ứng phó tình huống khẩn cấp và kiến tạo thành phố bền vững. Ứng dụng này cung cấp:

1. **Dữ liệu lịch sử**: Cho phép xem xu hướng tỷ lệ tử vong qua các năm
2. **Dự báo tương lai**: Sử dụng mô hình Prophet để dự báo tỷ lệ tử vong 5-6 năm tương lai
3. **Giao diện trực quan**: Bản đồ tương tác, biểu đồ thời gian, và bảng dữ liệu chi tiết

### Liên hệ với SDG 11.5

Dự án này hỗ trợ **Mục tiêu Phát triển Bền vững 11.5 (SDG 11.5)**: "Đến năm 2030, giảm đáng kể số người thiệt mạng do thảm họa, bao gồm cả thảm họa liên quan đến nước, và giảm thiệt hại kinh tế trực tiếp toàn cầu liên quan đến thảm họa". Ứng dụng cung cấp công cụ để theo dõi tiến độ và dự báo xu hướng này.

---

## 2. Công nghệ sử dụng

| Thành phần            | Công nghệ |
|-----------------------|-----------|
| **Backend**           | Flask |
| **Frontend**          | HTML5, Tailwind CSS, Chart.js |
| **Dự báo**            | Facebook Prophet, ARIMA |
| **Xử lý dữ liệu**     | Pandas, NumPy |
| **Phân tích**         | Scikit-learn, Statsmodels |
| **Visualize**         | Plotly, Matplotlib, Seaborn |
| **Bản đồ**            | Leaflet.js |
| **Icons**             | Font Awesome |
| **Định dạng dữ liệu** | CSV, JSON, Excel (XLSX) |

---

## 3. Cấu trúc thư mục

```
Mortality_Rate_Forecasting/
├── main.py                          # Entry point khởi chạy ứng dụng
├── requirements.txt                 # Danh sách thư viện Python
├── README.md                        # File hướng dẫn này
│
├── data/                            # Thư mục dữ liệu đầu vào
│   ├── time_series_country.csv      # Dữ liệu tỷ lệ tử vong lịch sử
│   ├── wb_population.csv            # Dữ liệu dân số từ World Bank
│   ├── emdat_disaster.xlsx          # Dữ liệu thảm họa từ EM-DAT
│   └── evaluate.xlsx                # Dữ liệu để đánh giá mô hình
│
├── saved/                           # Thư mục lưu mô hình và kết quả
│   ├── global.pkl                   # Mô hình toàn cầu
│   ├── region_*.pkl                 # Mô hình theo khu vực
│   ├── country_*.pkl                # Mô hình theo quốc gia
│   ├── forecast_data.csv            # Dữ liệu dự báo đã tính sẵn
│   └── metadata.json                # Thông tin metadata của các mô hình
│
├── src/                             # Source code chính
│   ├── model/                       # Các mô hình dự báo
│   │   ├── train_model.py           # Hệ thống huấn luyện mô hình
│   │   ├── prophet.py               # Lớp ProphetModel
│   │   └── arima.py                 # Lớp ARIMAModel
│   │
│   ├── processing/                  # Xử lý dữ liệu
│   │   └── aggregate.py             # Hàm tổng hợp dữ liệu theo level
│   │
│   ├── evaluation/                  # Đánh giá mô hình
│   │   └── split.py                 # Hàm chia dữ liệu expanding window
│   │
│   └── web/                         # Web application
│       ├── app.py                   # Flask app chính
│       ├── static/                  # Static files (CSS, JS)
│       │   ├── css/
│       │   │   └── style.css        # Stylesheet
│       │   └── js/
│       │       ├── charts.js        # Logic vẽ biểu đồ
│       │       ├── map.js           # Logic bản đồ Leaflet
│       │       └── ui.js            # Logic UI tương tác
│       │
│       └── templates/               # HTML templates
│           ├── index.html           # Trang chủ (bản đồ + thời gian trượt)
│           ├── about.html           # Trang giới thiệu dự án
│           ├── country_panel.html   # Panel chi tiết quốc gia
│           ├── forecast_modal.html  # Modal dự báo
│           ├── event_detail.html    # Chi tiết sự kiện
│           └── error.html           # Trang lỗi
│
├── notebook/                        # Jupyter notebooks (phân tích dữ liệu)
│   ├── eda.ipynb                    # Exploratory Data Analysis
│   ├── processing.ipynb             # Tiền xử lý dữ liệu
│   ├── arima.ipynb                  # Thực nghiệm mô hình ARIMA
│   ├── prophet.ipynb                # Thực nghiệm mô hình Prophet
│   ├── forecast.ipynb               # Dự báo tỷ lệ tử vong
│   └── evaluate.ipynb               # Đánh giá mô hình
│
├── examples/                        # Script ví dụ
│   ├── train_models_example.py      # Ví dụ huấn luyện mô hình
│   └── forecast_example.py          # Ví dụ tạo dự báo
│
└── docs/                            # Tài liệu bổ sung
```

#### Data Files
- **`time_series_country.csv`**: Dữ liệu chính
  - Cột: Country, Country Code (ISO3), Region, Subregion, Year, Total_Deaths, Population, Mortality_Rate
  - Phạm vi: 2000-2024

- **`forecast_data.csv`**: Dữ liệu dự báo tính sẵn
  - Cột: Country, Country Code, Region, Year, Mortality_Rate, yhat_lower, yhat_upper, Model_Source, Model_Path
  - Phạm vi dự báo: 2025-2030

---

## 4. Chức năng chính

### 1. Hiển thị Bản đồ Dữ liệu Theo Quốc gia
<img src="D:\Data\uel\HK7\Phân tích dữ liệu nâng cao\A3. SOURCES\Giao diện trang chủ.png"/>

### 3. Xem Chi tiết Quốc gia

<img src="D:\Data\uel\HK7\Phân tích dữ liệu nâng cao\A3. SOURCES\chi tiết quốc gia.png"/>

### 4. Hiển thị Dự báo

<img src="D:\Data\uel\HK7\Phân tích dữ liệu nâng cao\A3. SOURCES\dự báo.png"/>

### 5. Xuất Dữ liệu

- API `/api/export` hỗ trợ xuất:
  - **history**: Dữ liệu lịch sử theo quốc gia hoặc toàn cầu
  - **forecast**: Dữ liệu dự báo đến năm chỉ định
- Tên file: `historical_<COUNTRY_CODE>.csv` hoặc `forecast_<COUNTRY_CODE>_through_<YEAR>.csv`

---

## 5. Quy trình hoạt động

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUY TRÌNH HỆ THỐNG                         │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: CHUẨN BỊ DỮ LIỆU
├─ Nạp CSV từ sources
├─ Xử lý missing values, chuẩn hóa năm
└─ Tổng hợp theo 3 cấp độ: Global, Region, Country

PHASE 2: HUẤN LUYỆN MÔ HÌNH (offline, chạy 1 lần)
├─ Cấp độ Global: 1 mô hình Prophet trên toàn bộ dữ liệu
├─ Cấp độ Region: 1 mô hình Prophet cho mỗi 7 khu vực
└─ Cấp độ Country:
   ├─ Nếu ≥21 năm dữ liệu → train mô hình country
   ├─ Nếu 13-20 năm → fallback sang region model
   └─ Nếu <13 năm → không dự báo (NULL)

PHASE 3: TẠO DỮ LIỆU DỰ BÁO
├─ Load các mô hình .pkl từ saved/
├─ Chạy dự báo cho 6 năm tương lai (2025-2030)
├─ Lưu kết quả vào forecast_data.csv
└─ Lưu metadata.json (ghi lại model sources)

PHASE 4: KHỞI CHẠY WEB APPLICATION
├─ Flask nạp data CSV vào bộ nhớ
├─ Serve static files (CSS, JS) từ /static
├─ Render HTML templates từ /templates
└─ API ready to serve

PHASE 5: NGƯỜI DÙNG TƯƠNG TÁC
├─ User truy cập http://localhost:5000/
├─ Frontend gọi API:
│  ├─ GET /api/years → Danh sách năm có dữ liệu
│  ├─ GET /api/mortality-by-year/<year> → Dữ liệu năm
│  ├─ POST /api/forecast → Dự báo quốc gia
│  └─ GET /api/export → Download CSV
└─ Frontend render bản đồ, biểu đồ, panel
```
---

## 6. Hướng dẫn cài đặt

### Yêu cầu Môi trường

- **Python**: 3.8+
- **Hệ điều hành**: Windows, macOS, Linux

### Bước 1: Clone hoặc Tải Source Code

```bash
# Nếu sử dụng Git
git clone <repository-url>
cd Mortality_Rate_Forecasting

# Hoặc giải nén folder nếu tải ZIP
```

### Bước 2: Tạo Virtual Environment

```powershell
# Trên Windows
python -m venv venv
venv\Scripts\Activate.ps1

# Trên macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt Thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra Dữ liệu

Đảm bảo các file dữ liệu có sẵn trong thư mục `data/`:
- `time_series_country.csv` (dữ liệu lịch sử bắt buộc)
- `forecast_data.csv` nên có sẵn trong `saved/`

### Bước 5: Chạy Ứng dụng

```bash
python main.py
```

Output:
```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Debug mode: on
```

### Bước 6: Mở Trình Duyệt

- Truy cập: **http://localhost:5000/**
- Trang chủ sẽ tải với bản đồ tương tác

---

## 7. Ghi chú

### Nguồn Dữ liệu

- **Dữ liệu lịch sử tỷ lệ tử vong**: EM-DAT (Emergency Events Database)
- **Dữ liệu dân số**: World Bank (WB)
- **Phạm vi**: 2000-2024
- **Cấp độ**: Quốc gia, khu vực, toàn cầu

### Giới Hạn của Hệ Thống

1. **Dữ liệu Lịch sử**:
   - Chỉ có sẵn đến năm 2024
   - Một số quốc gia nhỏ/đặc biệt có thể không có đầy đủ dữ liệu

2. **Mô hình Dự báo**:
   - Sử dụng Prophet mà không có thông tin sự kiện tương lai
   - Dự báo chỉ dựa trên xu hướng lịch sử → có thể sai nếu có sự kiện đột biến
   - Quốc gia có <13 năm dữ liệu sẽ không có dự báo riêng (NULL values)

3. **Khoảng Tin cậy**:
   - Khoảng 95% dựa trên phân phối Prophet
   - Không đảm bảo 100% nằm trong khoảng (do bất định mô hình)

4. **Tính Chính xác**:
   - MAE, MAE, RMSE, Coverage
   - Expanding window evaluation được sử dụng

## 8. Đóng Góp

Chúng tôi hoan nghênh các đóng góp từ cộng đồng. Để đóng góp:

1. **Fork** repository
2. **Tạo branch** cho feature mới: `git checkout -b feature/your-feature-name`
3. **Commit** thay đổi: `git commit -am 'Add your feature'`
4. **Push** lên branch: `git push origin feature/your-feature-name`
5. **Tạo Pull Request** với mô tả chi tiết

### Hướng Dẫn Code

- Theo chuẩn PEP 8 cho Python
- Viết docstring cho hàm/class
- Cài đặt unit tests nếu thêm logic mới
- Update README nếu thay đổi cấu trúc

---

## 9. Liên Hệ

### Thông Tin Dự Án

- **Tên dự án**: Mortality Rate Forecasting System (Hệ thống Dự báo Tỷ lệ Tử vong)
- **Mục tiêu**: Hỗ trợ SDG 11.5 - Giảm thiệt hại do thảm họa
- **Trạng thái**: Đang phát triển

### Liên hệ

- Leader: Phạm Ngọc Mỹ
- Email: ngocmypzg@gmail.com
- Đơn vị: Trường Đại học Kinh tế - Luật, Đại học Quốc gia TP.HCM

### Hỗ Trợ & Báo Cáo Lỗi

- **GitHub Issues**: Tạo issue để báo cáo lỗi hoặc đề xuất tính năng
- **Email**: ngocmypzg@gmail.com
- **Documentation**: Xem thêm trong folder `/docs`

### Giấy Phép

- Dự án này được cấp phép dưới [MIT License](LICENSE)
---

*Cập nhật lần cuối: Tháng 3, 2026*

