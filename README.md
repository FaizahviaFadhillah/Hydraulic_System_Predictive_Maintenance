# Predictive-maintenance-system-hidrolik

Sistem prediktif pemeliharaan sistem hidraulik berbasis Machine Learning menggunakan algoritma **Histogram Gradient Boosting (HGB)**.

Proyek ini dirancang untuk memprediksi kondisi beberapa komponen utama pada sistem hidraulik berdasarkan data sensor, sehingga dapat membantu proses identifikasi kondisi komponen secara lebih dini.

## Tentang Proyek

Kegagalan pada komponen sistem hidraulik dapat mengganggu proses operasional dan meningkatkan kebutuhan pemeliharaan. Pendekatan predictive maintenance digunakan untuk memanfaatkan data sensor dan model Machine Learning dalam membantu memprediksi kondisi komponen.

Pada proyek ini, model **Histogram Gradient Boosting** digunakan untuk melakukan klasifikasi kondisi empat komponen:

- Cooler
- Valve
- Internal Pump Leakage
- Hydraulic Accumulator / Bar

Model dikembangkan melalui proses pengolahan data, preprocessing, pelatihan, dan evaluasi sebelum diterapkan ke dalam aplikasi berbasis Streamlit.

## Tujuan

- Membangun model Machine Learning untuk memprediksi kondisi komponen sistem hidraulik.
- Mengolah data sensor sebagai masukan model prediksi.
- Mengevaluasi performa model menggunakan Accuracy dan Macro F1-Score.
- Menerapkan model terlatih ke dalam aplikasi Streamlit untuk mempermudah penggunaan dan interpretasi hasil prediksi.

## Dataset

Dataset yang digunakan adalah **Condition Monitoring of Hydraulic Systems** yang berisi data sensor dari sistem uji hidraulik.

Data mencakup berbagai parameter pengukuran sistem yang digunakan sebagai fitur dalam proses prediksi kondisi komponen.

Target prediksi yang digunakan:

| Target | Kondisi yang Diprediksi |
|---|---|
| Cooler | Kondisi cooler |
| Valve | Kondisi valve |
| Internal Pump Leakage | Kondisi kebocoran internal pompa |
| Hydraulic Accumulator / Bar | Kondisi hydraulic accumulator |

## Alur Pengembangan

```text
Dataset
   ↓
Data Preprocessing
   ↓
Persiapan Fitur & Target
   ↓
Pembagian Data
   ↓
Pelatihan Model Histogram Gradient Boosting
   ↓
Evaluasi Model
   ↓
Penyimpanan Model
   ↓
Implementasi pada Streamlit
   ↓
Prediksi & Interpretasi Hasil
