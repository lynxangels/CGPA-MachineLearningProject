# 🎓 Analisis Performa Akademik Mahasiswa Melalui Pendekatan Machine Learning
> Studi Berbasis Data Kebiasaan dan Kondisi Sosial

Proyek ini membangun model **Machine Learning berbasis Regresi** untuk memprediksi **CGPA (Cumulative Grade Point Average)** mahasiswa berdasarkan kebiasaan belajar dan kondisi sosial-lingkungan — **tanpa menggunakan riwayat nilai akademik sebelumnya**.

---

## 📌 Latar Belakang

Sistem evaluasi akademik yang ada saat ini masih bersifat reaktif — penurunan performa baru terdeteksi setelah nilai turun signifikan. Padahal, kebiasaan sehari-hari mahasiswa seperti kehadiran, penggunaan media sosial, dan kondisi keluarga membentuk pola yang dapat diprediksi jauh sebelum masalah akademik muncul.

**Research Question:**
> *Apakah performa akademik mahasiswa dapat diprediksi berdasarkan kebiasaan dan kondisi hidup — tanpa menunggu nilai akhir keluar?*

---

## 🗂️ Dataset

| Atribut | Detail |
|---|---|
| Sumber | Survei mahasiswa BCSE, universitas swasta, Bangladesh |
| Referensi | Hasan et al. (2024), [Mendeley Data](https://doi.org/10.17632/dc3797vf3t/1) |
| Jumlah Data | 1.194 mahasiswa |
| Total Variabel | 31 (raw) → 22 fitur + 1 target (setelah seleksi) |
| Target (Y) | CGPA skala 0.0 – 4.0 |
| Missing Value | Tidak ada |

### Fitur Input (22 Variabel)

**Main Variables — Kebiasaan Belajar (9 fitur)**
- Jam belajar harian, frekuensi sesi belajar, jam skill development
- Jam penggunaan media sosial, tingkat kehadiran, mode belajar
- Status probation, konsultasi dosen, keterlibatan ko-kurikuler

**Support Variables — Faktor Lingkungan (13 fitur)**
- Current semester, usia, jenis kelamin, status beasiswa
- Pendapatan keluarga bulanan, profisiensi bahasa Inggris
- Status tinggal, kondisi kesehatan, kepemilikan perangkat, dll.

---

## ⚙️ Tech Stack

```
pandas · numpy · scipy · seaborn · matplotlib
scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)
streamlit · joblib · ngrok
```

---

## 🧠 Metodologi

### 1. Preprocessing Pipeline (Zero Data Leakage)
Seluruh preprocessing di-fit **hanya dari X_train** menggunakan `scikit-learn Pipeline`:

```
ColumnTransformer
├── log_num   → log1p transform + StandardScaler  (6 fitur skewed)
├── plain_num → StandardScaler saja               (2 fitur: Age, Semester)
└── cat       → OneHotEncoder (drop='first')      (14 fitur kategorikal)
```

### 2. Model yang Diuji

| Model | Keterangan |
|---|---|
| Linear Regression | Baseline |
| Lasso Regression | alpha=0.01, seleksi fitur otomatis |
| Ridge Regression | alpha=1.0, regularisasi L2 |
| **Random Forest** | **Model Utama**, 100–200 trees |

### 3. Evaluasi
- **5-Fold Cross-Validation** pada training set
- **GridSearchCV** untuk tuning hyperparameter Random Forest
- Metrik: **MAE**, **RMSE**, **R²**

### 4. Hasil Cross-Validation

| Model | CV R² | CV MAE | CV RMSE |
|---|---|---|---|
| Linear Regression | 0.0528 ± 0.055 | 0.4886 | 0.7338 |
| Lasso Regression | 0.0626 ± 0.034 | 0.4859 | 0.7304 |
| Ridge Regression | 0.0542 ± 0.053 | 0.4881 | 0.7333 |
| Random Forest | 0.2620 ± 0.171 | 0.4397 | 0.6421 |
| **RF (Tuned)** ✅ | **0.2910** | **0.443** | **—** |

### 5. Best Hyperparameters (GridSearchCV)

```python
{
  'n_estimators':    200,
  'max_depth':       10,
  'min_samples_split': 5,
  'min_samples_leaf':  2
}
```

---

## 📊 Key Findings

- **Current Semester** adalah fitur paling berpengaruh (importance = **0.350**) — efek senioritas/akumulasi akademik
- **Faktor Lingkungan (65.6%)** secara kolektif lebih prediktif dibanding Kebiasaan Belajar (34.4%)
- **Kehadiran di kelas (0.086)** adalah kebiasaan paling prediktif — mengalahkan jam belajar harian
- **Jam media sosial (0.057)** berkorelasi negatif dengan CGPA
- **Monthly Family Income** hampir tidak berkorelasi linear (r = −0.0003) namun menjadi fitur penting di RF — bukti hubungan non-linear dalam data
- Lasso mengeliminasi **13 dari 26 fitur** terenkoding, termasuk jam belajar harian dan kepemilikan gadget

---

## 🚀 Deployment

Aplikasi di-deploy menggunakan **Streamlit** yang berjalan di **Google Colab** dan diekspos ke publik via **ngrok**.

```
User (Browser)
    │ HTTPS Request
    ▼
ngrok Tunnel (Public URL)
    │ Forward → localhost:8501
    ▼
Streamlit UI (22 Input Form)
    │ User Submit
    ▼
Preprocessing Pipeline (log1p + Scaler + OHE)
    │
    ▼
Random Forest Regressor (loaded via joblib)
    │
    ▼
Prediksi CGPA (0.0 – 4.0)
    │
    ▼
Output: Nilai + Kategori + Rekomendasi + Feature Importance
```

### Kategori Output

| Kategori | Rentang CGPA |
|---|---|
| 🏆 Performa Tinggi | ≥ 3.5 |
| 📘 Performa Sedang | 2.75 – 3.49 |
| ⚠️ Perlu Perhatian | < 2.75 |

---

## 📁 Struktur Proyek

```
├── app.py                  # Streamlit application
├── model/
│   └── best_model.pkl      # Trained Random Forest pipeline (joblib)
├── notebook/
│   └── ML_Final_Project.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ Cara Menjalankan

### Lokal
```bash
# Clone repo
git clone https://github.com/lynxangels/CGPA-MachineLearningProject.git
cd student-cgpa-prediction

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

### Via Google Colab + ngrok
```python
# Di Colab, jalankan cell berikut:
!pip install streamlit pyngrok
from pyngrok import ngrok
!streamlit run app.py &
public_url = ngrok.connect(8501)
print(public_url)
```

---

## ⚠️ Limitasi

1. **Distribusi tidak merata** — mahasiswa CGPA rendah (< 2.75) adalah minoritas, model kurang optimal untuk kelompok berisiko
2. **Single institution** — data hanya dari satu universitas di Bangladesh, generalisasi terbatas
3. **Self-reported data** — rentan social desirability bias, tidak ada validasi objektif
4. **Deployment sementara** — Google Colab + ngrok tidak stabil untuk produksi jangka panjang

---

## 👩‍💻 Tim

| Nama | NIM |
|---|---|
| Jesselyn Angelia Luwuk | 2802420252 |
| Maria Yohana Vianny Leo | 2802538813 |
| Prabandari Pramesti Larasati Putri | 2802514403 |

**Universitas Bina Nusantara — 2026/2027**

---

## 📚 Referensi

- Hasan, T., Hasan, M. M., & Manzoor, T. (2024). *Student Performance Metrics Dataset*. Mendeley Data. https://doi.org/10.17632/dc3797vf3t/1
- Batool, S., et al. (2023). Educational data mining to predict students' academic performance: A survey study. *Education and Information Technologies*, 28(1), 905–971.
- Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict secondary school student performance. *FUBUTEC 2008*, 5–12.
