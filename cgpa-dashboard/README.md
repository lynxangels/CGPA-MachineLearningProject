# Prediksi CGPA Mahasiswa (Vercel)

Struktur:
- `public/index.html`  : tampilan dashboard
- `api/predict.py`     : endpoint prediksi (Python serverless)
- `best_model.pkl`     : TARUH FILE MODEL KAMU DI SINI (folder paling luar)
- `requirements.txt`   : ganti versi scikit-learn sesuai Colab

Langkah deploy:
1. Salin `best_model.pkl` dari Colab ke folder ini (sejajar dengan `vercel.json`).
2. Di Colab jalankan `import sklearn; print(sklearn.__version__)`, lalu isi di `requirements.txt` (mis. `scikit-learn==1.6.1`).
3. Upload folder ini ke repo GitHub baru.
4. Di vercel.com: Add New > Project > pilih repo > Framework Preset: Other > Deploy.
5. Buka link Vercel-nya dan coba prediksi.

Jangan sertakan token ngrok atau file notebook yang berisi token ke GitHub.
