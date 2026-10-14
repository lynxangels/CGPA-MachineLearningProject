# Prediksi CGPA Mahasiswa (Vercel)

- `public/index.html` : tampilan dashboard (yang dilihat pengunjung)
- `api/predict.py`    : endpoint prediksi (Python serverless), murni Python tanpa scikit-learn
- `model.json`        : model Random Forest hasil ekspor dari best_model.pkl

Deploy:
1. Upload folder ini ke repo GitHub baru.
2. vercel.com > Add New > Project > pilih repo > Framework Preset: Other > Deploy.
3. Buka link-nya dan coba prediksi.

Jangan sertakan token ngrok atau notebook yang berisi token ke GitHub.
