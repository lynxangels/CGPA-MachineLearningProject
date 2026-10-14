import streamlit as st
import pandas as pd
import numpy as np
import pickle, os

st.set_page_config(
    page_title="Prediksi CGPA Mahasiswa",
    page_icon="🎓",
    layout="wide",
)

st.markdown('''
<style>
.main-header{background:linear-gradient(135deg,#1565C0,#1976D2);color:white;
  padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;}
.main-header h1{color:white;margin:0;font-size:1.6rem;}
.main-header p{color:#BBDEFB;margin:.3rem 0 0;font-size:.9rem;}
.result-high{background:#F1F8E9;border:2px solid #66BB6A;border-radius:12px;
  padding:1.5rem;text-align:center;margin-bottom:1rem;}
.result-mid{background:#FFF8E1;border:2px solid #FFA726;border-radius:12px;
  padding:1.5rem;text-align:center;margin-bottom:1rem;}
.result-low{background:#FFEBEE;border:2px solid #EF5350;border-radius:12px;
  padding:1.5rem;text-align:center;margin-bottom:1rem;}
.cgpa-high{font-size:4rem;font-weight:700;color:#2E7D32;line-height:1;}
.cgpa-mid{font-size:4rem;font-weight:700;color:#E65100;line-height:1;}
.cgpa-low{font-size:4rem;font-weight:700;color:#B71C1C;line-height:1;}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pkl"):
        st.error("File best_model.pkl tidak ditemukan, Coba muat ulang")
        st.stop()
    with open("best_model.pkl","rb") as f:
        return pickle.load(f)

model = load_model()

st.markdown("""
<div class="main-header">
  <h1>🎓 Prediksi CGPA Mahasiswa</h1>
  <p>Analisis Performa Akademik berbasis Kebiasaan Belajar & Faktor Lingkungan — Model: Random Forest</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📋 Input Data Mahasiswa")
    st.markdown("---")
    st.markdown("### 📚 Kebiasaan Belajar")

    study_hours    = st.slider("Jam belajar per hari", 0.0, 12.0, 5.0, 0.5)
    study_sessions = st.slider("Frekuensi sesi belajar per hari", 0, 10, 3, 1)
    social_media   = st.slider("Jam media sosial per hari", 0.0, 10.0, 2.0, 0.5)
    skill_dev      = st.slider("Jam pengembangan skill per hari", 0.0, 8.0, 1.0, 0.5)
    attendance     = st.slider("Rata-rata kehadiran kelas (%)", 0, 100, 80, 1)
    learning_mode  = st.selectbox("Mode belajar", ["Online","Offline","Blended"], index=1)
    probation      = st.selectbox("Pernah probation?", ["No","Yes"])
    teacher_consult= st.selectbox("Konsultasi akademik dengan dosen?", ["Yes","No"])
    co_curriculum  = st.selectbox("Aktif co-kurikuler?", ["Yes","No"])

    st.markdown("---")
    st.markdown("### 🌍 Faktor Lingkungan")

    gender         = st.selectbox("Gender", ["Male","Female"])
    age            = st.slider("Usia", 17, 35, 21, 1)
    semester       = st.selectbox("Semester", ["1","2","3","4","5","6","7","8"], index=2)
    scholarship    = st.selectbox("Beasiswa meritorious?", ["No","Yes"])
    english_prof   = st.selectbox("Profisiensi bahasa Inggris", ["Basic","Intermediate","Advanced"], index=1)
    living_with    = st.selectbox("Tinggal bersama?", ["Family","Friends","Alone"])
    health_issue   = st.selectbox("Masalah kesehatan?", ["No","Yes"])
    family_income  = st.number_input("Pendapatan keluarga/bulan (angka)",min_value=0,max_value=50000000,value=50000, step=1000,
    help="Masukkan nominal pendapatan keluarga per bulan. Contoh: 3000000")
    personal_pc    = st.selectbox("Punya komputer/laptop pribadi?", ["Yes","No"])
    relationship   = st.selectbox("Status hubungan", ["Single","In Relationship"])
    smartphone     = st.selectbox("Menggunakan smartphone?", ["Yes","No"])
    disability     = st.selectbox("Disabilitas fisik?", ["No","Yes"])
    transportation = st.selectbox("Transportasi kampus?", ["Yes","No"])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict CGPA", use_container_width=True, type="primary")

if predict_btn:
    input_data = pd.DataFrame([{
        "How many hour do you study daily?"                                          : study_hours,
        "How many times do you seat for study in a day?"                            : float(study_sessions),
        "How many hour do you spent daily in social media?"                         : social_media,
        "How many hour do you spent daily on your skill development?"               : skill_dev,
        "Average attendance on class"                                               : float(attendance),
        "What is your preferable learning mode?"                                    : learning_mode,
        "Did you ever fall in probation?"                                           : probation,
        "Do you attend in teacher consultancy for any kind of academical problems?" : teacher_consult,
        "Are you engaged with any co-curriculum activities?"                        : co_curriculum,
        "Gender"                                                                    : gender,
        "Age"                                                                       : float(age),
        "Current Semester"                                                          : float(semester),
        "Do you have meritorious scholarship ?"                                     : scholarship,
        "Status of your English language proficiency"                               : english_prof,
        "With whom you are living with?"                                            : living_with,
        "Do you have any health issues?"                                            : health_issue,
        "What is your monthly family income?"                                       : float(family_income),
        "Do you have personal Computer?"                                            : personal_pc,
        "What is your relationship status?"                                         : relationship,
        "Do you use smart phone?"                                                   : smartphone,
        "Do you have any physical disabilities?"                                    : disability,
        "Do you use University transportation?"                                     : transportation,
    }])

    try:
        cgpa_pred = float(model.predict(input_data)[0])
        cgpa_pred = round(min(4.0, max(0.0, cgpa_pred)), 2)
    except Exception as e:
        st.error(f"Gagal prediksi: {e}")
        st.stop()

    pct = round((cgpa_pred / 4.0) * 100)

    if cgpa_pred >= 3.5:
        cat="Performa Tinggi"; emoji="🏆"; box="result-high"; num="cgpa-high"
        color="#2E7D32"
        interp=f"Mahasiswa ini menunjukkan **performa akademik sangat baik** (CGPA {cgpa_pred:.2f})."
        tips=["Pertahankan konsistensi belajar","Optimalkan skill development","Jadilah mentor bagi teman"]
        icon="✅"
    elif cgpa_pred >= 2.75:
        cat="Performa Sedang"; emoji="📘"; box="result-mid"; num="cgpa-mid"
        color="#E65100"
        interp=f"Mahasiswa ini pada **jalur yang cukup baik** (CGPA {cgpa_pred:.2f}), namun masih bisa ditingkatkan."
        tips=["Tingkatkan jam belajar harian","Kurangi waktu media sosial","Rutin konsultasi dosen"]
        icon="⚡"
    else:
        cat="Perlu Perhatian"; emoji="⚠️"; box="result-low"; num="cgpa-low"
        color="#B71C1C"
        interp=f"Performa akademik **(CGPA {cgpa_pred:.2f}) memerlukan perhatian segera**."
        tips=["Segera hubungi dosen pembimbing","Atur ulang jadwal belajar","Kurangi kegiatan non-akademik"]
        icon="🚨"

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown(f"""
        <div class="{box}">
          <div style="font-size:.8rem;color:#757575;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">Prediksi CGPA</div>
          <div class="{num}">{cgpa_pred:.2f}</div>
          <div style="color:#9E9E9E;font-size:.8rem;margin-top:4px;">dari 4.00 (skala maksimum)</div>
          <div style="background:#E0E0E0;border-radius:999px;height:8px;margin:14px 0 10px;overflow:hidden;">
            <div style="width:{pct}%;height:8px;border-radius:999px;background:{color};"></div>
          </div>
          <b>{emoji} {cat}</b>
        </div>
        """, unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("CGPA", f"{cgpa_pred:.2f}")
        m2.metric("Persentil", f"{pct}%")
        m3.metric("Kategori", cat.split()[0])

    with col2:
        st.markdown("#### 📊 Interpretasi")
        if cgpa_pred >= 3.5:   st.success(interp)
        elif cgpa_pred >= 2.75: st.warning(interp)
        else:                  st.error(interp)

        st.markdown("#### 💡 Rekomendasi")
        for t in tips:
            st.markdown(f"{icon} {t}")

    st.markdown("---")
    st.markdown("### 📋 Ringkasan Input")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kebiasaan Belajar**")
        st.dataframe(pd.DataFrame({
            "Variabel":["Jam belajar","Sesi belajar","Media sosial","Skill dev",
                        "Kehadiran","Mode belajar","Probation","Konsultasi","Ko-kurikuler"],
            "Nilai":[f"{study_hours} jam",f"{study_sessions} sesi",f"{social_media} jam",
                     f"{skill_dev} jam",f"{attendance}%",learning_mode,
                     probation,teacher_consult,co_curriculum],
        }), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Faktor Lingkungan**")
        st.dataframe(pd.DataFrame({
            "Variabel":["Gender","Usia","Semester","Beasiswa","Bahasa Inggris",
                        "Tinggal bersama","Kesehatan","Pendapatan","Komputer",
                        "Hubungan","Smartphone","Disabilitas","Transportasi"],
            "Nilai":[gender,f"{age} thn",semester,scholarship,english_prof,
                     living_with,health_issue,family_income,personal_pc,
                     relationship,smartphone,disability,transportation],
        }), use_container_width=True, hide_index=True)

else:
    st.info("👈 Isi variabel di sidebar lalu tekan **Predict CGPA**", icon="ℹ️")
    with st.expander("ℹ️ Tentang Model"):
        st.markdown("""
        **Model:** Random Forest Regressor (GridSearchCV Tuned)\n
        **Variabel:** 9 Main (Kebiasaan) + 13 Support (Lingkungan) = 22 total\n
        **Target:** CGPA 0.0 – 4.0
        **Kategori:** Tinggi (≥3.5) | Sedang (2.75–3.49) | Perlu Perhatian (<2.75)
        """)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#9E9E9E;font-size:.8rem;'>Final Project — Analisis Performa Akademik | Random Forest</div>", unsafe_allow_html=True)
