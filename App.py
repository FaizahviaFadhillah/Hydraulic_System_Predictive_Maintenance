"""
Aplikasi Streamlit 
- Upload model .pkl (Cooler / Valve / PumpLeakage / Accumulator) joblib.dump(...)
- Upload CSV fitur (kolom sama seperti saat training / prediksi.csv)
- Prediksi -> tabel + interpretasi + unduh CSV
"""

import io
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ====================== Konfigurasi dasar ======================
st.set_page_config(page_title="Prediksi Maintenance — Hidraulik ", layout="centered")
st.title("🔧 Prediksi Maintenance Hidraulik")
st.caption(
    "Upload model, upload data, lalu jalankan prediksi. "
    "Data fitur diasumsikan sudah melalui tahapan prapemrosesan di notebook."
)

# Pengaturan tampilan (dikunci, tidak dari sidebar)
tampilkan_tabel_angka = True
tampilkan_prob = False
fill_missing_default = 0.0  # nilai default untuk fitur yang hilang (mis. Stable_flag)

# ====================== Sidebar: upload artefak ======================
st.sidebar.header("1) Upload Artefak Model (.pkl)")
st.sidebar.write("Gunakan file .pkl yang disimpan dengan **joblib.dump(...)** dari notebook.")

up_X1 = st.sidebar.file_uploader("Upload cooler_model.pkl",      type=["pkl"])
up_X2 = st.sidebar.file_uploader("Upload valve_model.pkl",       type=["pkl"])
up_X3 = st.sidebar.file_uploader("Upload pump_model.pkl",        type=["pkl"])
up_X4 = st.sidebar.file_uploader("Upload accumulator_model.pkl", type=["pkl"])

st.sidebar.divider()
st.sidebar.header("2) Upload Data Fitur (CSV)")
up_csv = st.sidebar.file_uploader(
    "Pilih file CSV (kolom & nama sama seperti saat training / prediksi.csv)",
    type=["csv"],
)

# ====================== Loader model ======================
def load_model(file) -> Optional[object]:
    if file is None:
        return None
    try:
        buf = io.BytesIO(file.read())
        return joblib.load(buf)
    except Exception as e:
        st.error(f"Gagal membaca file model (.pkl): {e}")
        return None


model_X1 = load_model(up_X1)
model_X2 = load_model(up_X2)
model_X3 = load_model(up_X3)
model_X4 = load_model(up_X4)

missing = [
    n
    for n, o in {
        "Cooler": model_X1,
        "Valve": model_X2,
        "PumpLeakage": model_X3,
        "Accumulator": model_X4,
    }.items()
    if o is None
]

if missing:
    st.warning(
        "Model belum lengkap: " + ", ".join(missing) +
        ". Prediksi hanya dijalankan untuk model yang tersedia."
    )
else:
    st.success("Semua model sudah ter-upload.")

# ====================== Data CSV ======================
st.subheader("Langkah Penggunaan")
st.markdown("1) Upload **model (.pkl)** • 2) Upload **CSV fitur** • 3) Klik **Jalankan Prediksi**")

X: Optional[pd.DataFrame] = None
if up_csv is not None:
    try:
        X = pd.read_csv(up_csv)
        st.write("**Data terbaca** (5 baris pertama):")
        st.dataframe(X.head())
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")

st.divider()
st.subheader("Jalankan Prediksi")
run = st.button("Jalankan Prediksi", type="primary")

# ====================== Utilitas ======================
def align_columns(df: pd.DataFrame, model) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Samakan urutan kolom dengan model.
    Jika ada fitur hilang, otomatis dibuat dan diisi nilai default (fill_missing_default).
    """
    try:
        # Cek metadata kustom dulu, lalu fallback ke sklearn
        feat = getattr(model, "_feature_names", None)
        if feat is None:
            feat = getattr(model, "feature_names_in_", None)
        if feat is None:
            return df, None

        feat = list(feat)
        missing = [c for c in feat if c not in df.columns]
        extra = [c for c in df.columns if c not in feat]

        warn_parts = []
        if missing:
            for c in missing:
                df[c] = fill_missing_default
            warn_parts.append(
                f"kolom hilang: {missing} -> dibuat & diisi {fill_missing_default}"
            )
        if extra:
            warn_parts.append(f"kolom ekstra: {extra} (diabaikan)")

        warn = (
            "Tidak semua kolom cocok dengan model (" + "; ".join(warn_parts) + ")."
            if warn_parts
            else None
        )
        df2 = df[[c for c in feat]]  # urut sesuai training
        return df2, warn
    except Exception as e:
        return df, f"Gagal menyamakan kolom: {e}"


def predict_series(name: str, model, features: pd.DataFrame):
    """
    Prediksi satu komponen:
    - model.predict -> index kelas (0..K-1)
    - decode ke label asli pakai model.label_mapping_ kalau tersedia
    """
    if model is None:
        return None, None, None
    try:
        X_use, warn = align_columns(features, model)
        if warn:
            st.warning(f"{name}: {warn}")

        # Prediksi index kelas
        yhat_idx = model.predict(X_use.values)

        # Decode ke label asli (100, 20, 3, 90, 80, 73, 130, dst.)
        mapping = getattr(model, "label_mapping_", None)
        if mapping is not None:
            inv_map = {v: k for k, v in mapping.items()}  # index -> label asli
            yhat = np.array([inv_map.get(int(v), v) for v in yhat_idx])
        else:
            yhat = yhat_idx

        y_ser = pd.Series(yhat, index=features.index, name=f"{name}_pred")

        # Opsional: probabilitas
        proba_df = None
        if tampilkan_prob and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_use.values)
                proba_df = pd.DataFrame(proba, index=features.index)
                proba_df.columns = [f"{name}_p{c}" for c in range(proba_df.shape[1])]
            except Exception:
                proba_df = None

        return y_ser, proba_df, None

    except Exception as e:
        return None, None, f"Prediksi untuk {name} gagal: {e}"


# ======= Fungsi interpretasi: pakai LABEL ASLI, bukan index =======
def msg_cooler(v: float) -> str:
    if v == 100:
        return "✅ Cooler: Full efficiency. Tidak diperlukan tindakan pemeliharaan."
    if v == 20:
        return "🟠 Cooler: Reduced efficiency. Efisiensi pendinginan menurun, perlu dijadwalkan pemeliharaan."
    if v == 3:
        return "⚠️ Cooler: Close to total failure. Pendingin mendekati kegagalan total, perlu tindakan segera."
    return "⚠️ Cooler: Label kondisi tidak dikenali."


def msg_valve(v: float) -> str:
    if v == 100:
        return "✅ Valve: Optimal switching behavior. Perpindahan posisi cepat dan stabil."
    if v == 90:
        return "🟡 Valve: Small lag. Terdapat sedikit keterlambatan switching, perlu pemantauan."
    if v == 80:
        return "🟠 Valve: Severe lag. Keterlambatan switching signifikan, jadwalkan pemeliharaan."
    if v == 73:
        return "⚠️ Valve: Close to total failure. Valve hampir gagal berfungsi, risiko gangguan sistem tinggi."
    return "⚠️ Valve: Label kondisi tidak dikenali."


def msg_pump(v: float) -> str:
    if v == 0:
        return "✅ Pump: No leakage. Tidak terdeteksi kebocoran internal."
    if v == 1:
        return "🟡 Pump: Weak leakage. Kebocoran ringan, indikasi awal kerusakan, rencanakan perbaikan."
    if v == 2:
        return "⚠️ Pump: Severe leakage. Kebocoran berat, pompa berisiko gagal dan perlu tindakan segera."
    return "⚠️ Pump: Label kondisi tidak dikenali."


def msg_acc(v: float) -> str:
    if v == 130:
        return "✅ Accumulator: Optimal pressure. Tekanan berada pada rentang optimal."
    if v == 115:
        return "🟡 Accumulator: Slightly reduced pressure. Tekanan sedikit menurun, indikasi degradasi awal."
    if v == 100:
        return "🟠 Accumulator: Severely reduced pressure. Tekanan jauh di bawah optimal, kapasitas simpan energi menurun."
    if v == 90:
        return "⚠️ Accumulator: Close to total failure. Tekanan sangat rendah, akumulator hampir tidak berfungsi dengan baik."
    return "⚠️ Accumulator: Label kondisi tidak dikenali."


def ringkas_baris(row: pd.Series) -> str:
    parts = []
    for col in ["Cooler_info", "Valve_info", "Pump_info", "Accumulator_info"]:
        if col in row and pd.notna(row[col]):
            txt = (
                str(row[col])
                .replace("✅ ", "")
                .replace("🟡 ", "")
                .replace("🟠 ", "")
                .replace("⚠️ ", "")
            )
            parts.append(txt)

    if not parts:
        return "Tidak ada informasi prediksi."

    # Urutkan berdasarkan tingkat keparahan
    def get_priority(text):
        text = text.lower()
        if "segera" in text or "kritis" in text or "gagal" in text:
            return 0  # paling penting
        elif "severe" in text or "berat" in text:
            return 1
        elif "reduced" in text or "lag" in text or "menurun" in text:
            return 2
        else:
            return 3  # normal

    parts_sorted = sorted(parts, key=get_priority)

    # Ambil maksimal 4 biar semua komponen tetap tampil
    return " • ".join(parts_sorted[:4])


# ====================== Eksekusi prediksi ======================
if run:
    if X is None:
        st.error("Silakan upload CSV terlebih dahulu di sidebar.")
    else:
        Xp = X.copy()

        pred_cols: Dict[str, pd.Series] = {}
        proba_cols: Dict[str, pd.DataFrame] = {}
        errors = []

        for name, mdl in [
            ("Cooler", model_X1),
            ("Valve", model_X2),
            ("PumpLeakage", model_X3),
            ("Accumulator", model_X4),
        ]:
            y_ser, proba_df, err = predict_series(name, mdl, Xp)
            if err:
                errors.append(err)
            if y_ser is not None:
                pred_cols[f"{name}_pred"] = y_ser
            if proba_df is not None:
                proba_cols[name] = proba_df

        for e in errors:
            st.error(e)

        if not pred_cols:
            st.error(
                "Tidak ada model yang berhasil dipakai. "
                "Pastikan minimal satu model ter-upload dan kolom CSV sesuai."
            )
        else:
            out = pd.DataFrame(pred_cols, index=Xp.index)
            st.success("Prediksi selesai.")

            if tampilkan_tabel_angka:
                st.dataframe(out)

            # Interpretasi teks
            out_text = pd.DataFrame(index=out.index)
            if "Cooler_pred" in out:
                out_text["Cooler_info"] = out["Cooler_pred"].apply(msg_cooler)
            if "Valve_pred" in out:
                out_text["Valve_info"] = out["Valve_pred"].apply(msg_valve)
            if "PumpLeakage_pred" in out:
                out_text["Pump_info"] = out["PumpLeakage_pred"].apply(msg_pump)
            if "Accumulator_pred" in out:
                out_text["Accumulator_info"] = out["Accumulator_pred"].apply(msg_acc)

            st.write("---")
            st.subheader("Interpretasi")
            st.dataframe(out_text)

            if not out_text.empty:
                st.write("---")
                st.subheader("Ringkasan singkat (1 kalimat, baris pertama)")
                st.success(ringkas_baris(out_text.iloc[0]))

            if not out_text.empty:
                summary_df = pd.DataFrame(
                    {"summary": [ringkas_baris(r) for _, r in out_text.iterrows()]}
                )
                buf_sum = io.StringIO()
                summary_df.to_csv(buf_sum, index=False)
                st.download_button(
                    "⬇️ Unduh Ringkasan (CSV)",
                    buf_sum.getvalue(),
                    "ringkasan_prediksi.csv",
                    "text/csv",
                )

            # Unduh gabungan input + prediksi + interpretasi
            buf = io.StringIO()
            pd.concat([X, out, out_text], axis=1).to_csv(buf, index=False)
            st.download_button(
                "⬇️ Unduh Hasil (CSV)",
                buf.getvalue(),
                "hasil_prediksi.csv",
                "text/csv",
            )

st.info(
    "Catatan: File .pkl harus disimpan dengan joblib.dump(...). "
    "Data CSV (misalnya prediksi.csv) diasumsikan sudah diproses "
    "dengan tahapan prapemrosesan yang sama seperti saat pelatihan model."
)
