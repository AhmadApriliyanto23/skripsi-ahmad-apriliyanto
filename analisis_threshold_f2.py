"""
=====================================================================
ANALISIS THRESHOLD KATEGORI RISIKO (MODEL HASIL TUNING F2)
=====================================================================
WAJIB DIULANG -- distribusi probabilitas berubah setelah tuning ulang
dengan F2-score.

PENTING: Kode ini memakai model_xgb sebagai model final. Cek dulu hasil
batch5_evaluasi_f2.py -- jika model terpilih ternyata Random Forest,
ganti baris `model = joblib.load(...)` di bawah menjadi
"model_rf_final_f2.pkl".

PRASYARAT: batch3_tuning_f2.py dan batch4_feature_importance_f2.py
sudah dijalankan.
=====================================================================
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

print("\n" + "#" * 70)
print("# ANALISIS THRESHOLD KATEGORI RISIKO (MODEL HASIL TUNING F2)")
print("#" * 70)
print(">>> GANTI model di bawah jika model terpilih bukan XGBoost <<<")
print(">>> (cek hasil Batch 5 terlebih dahulu)                    <<<")

model = joblib.load("model_xgb_final_f2.pkl")  # sesuaikan jika model terpilih berbeda
X_test = pd.read_csv("X_test_f2.csv")
y_test = pd.read_csv("y_test_f2.csv").squeeze("columns")

proba_final = model.predict_proba(X_test)[:, 1]

print("\n=== STATISTIK DISTRIBUSI PROBABILITAS (prop_no_show) ===")
print(pd.Series(proba_final).describe())

base_rate = y_test.mean()
print(f"\nProporsi aktual no-show pada data uji (base rate): {base_rate:.4f} ({base_rate * 100:.2f}%)")

print("\n=== TABEL TRADE-OFF PER THRESHOLD ===")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'F2':>10} {'% Tinggi':>12}")
for t in np.arange(0.10, 0.95, 0.05):
    y_pred_t = (proba_final >= t).astype(int)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    f2 = fbeta_score(y_test, y_pred_t, beta=2, zero_division=0)
    pct_flagged = (y_pred_t == 1).mean() * 100
    print(f"{t:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {f2:>10.4f} {pct_flagged:>11.2f}%")

p_tinggi = np.percentile(proba_final, 100 - (base_rate * 100))
p_sedang = np.percentile(proba_final, 50)

print(f"\n=== USULAN THRESHOLD BERBASIS PERSENTIL (base rate {base_rate * 100:.1f}%) ===")
print(f"Threshold kategori TINGGI: {p_tinggi:.4f}")
print(f"Threshold kategori SEDANG (median): {p_sedang:.4f}")
print(f"\nUsulan pembagian 3 kategori:")
print(f"  Rendah : prop_no_show < {p_sedang:.4f}")
print(f"  Sedang : {p_sedang:.4f} <= prop_no_show < {p_tinggi:.4f}")
print(f"  Tinggi : prop_no_show >= {p_tinggi:.4f}")
print("\n>>> THRESHOLD INI PERLU DIPAKAI ULANG DI .env FASTAPI (ganti nilai lama) <<<")

print("\n>>> LANJUT KE: batch6_shap_f2.py <<<")
