"""
=====================================================================
BATCH 6 (REVISI) : INTERPRETASI SHAP (MODEL HASIL TUNING F2)
=====================================================================
Asumsi model terpilih tetap XGBoost. Sesuaikan baris joblib.load(...)
di bawah jika model terpilih ternyata berbeda (cek hasil Batch 5).

PRASYARAT: batch3_tuning_f2.py dan batch4_feature_importance_f2.py
sudah dijalankan (butuh model_xgb_final_f2.pkl, X_test_f2.csv).

OUTPUT FILE:
- shap_summary_plot_f2.png
- shap_bar_plot_f2.png
- shap_feature_ranking_f2.csv
- shap_force_plot_positif_f2.png
- shap_force_plot_negatif_f2.png
=====================================================================
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from utils_fitur import fitur

print("\n" + "#" * 70)
print("# BATCH 6 (REVISI) : INTERPRETASI SHAP (MODEL HASIL TUNING F2)")
print("#" * 70)
print(">>> Kode ini asumsi model terpilih tetap XGBoost. Sesuaikan jika berbeda <<<")

model_xgb = joblib.load("model_xgb_final_f2.pkl")  # sesuaikan jika model terpilih berbeda
X_test = pd.read_csv("X_test_f2.csv")

np.random.seed(42)
sample_size = min(2000, len(X_test))
sample_idx = np.random.choice(X_test.index, size=sample_size, replace=False)
X_sample = X_test.loc[sample_idx]
print(f"Jumlah sampel yang digunakan untuk SHAP: {len(X_sample)}")

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_sample, check_additivity=False)
print(f"Shape SHAP values: {shap_values.shape}")

plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=fitur, show=False)
plt.title("SHAP Summary Plot - Prediksi No-Show (XGBoost, F2-tuned)")
plt.tight_layout()
plt.savefig("shap_summary_plot_f2.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=fitur, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (F2-tuned)")
plt.tight_layout()
plt.savefig("shap_bar_plot_f2.png", dpi=150, bbox_inches='tight')
plt.close()

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'fitur': fitur, 'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)
print("\n=== RANKING FITUR BERDASARKAN SHAP (GLOBAL, F2-tuned) ===")
print(shap_importance.to_string(index=False))
shap_importance.to_csv("shap_feature_ranking_f2.csv", index=False)

proba_sample = model_xgb.predict_proba(X_sample)[:, 1]
idx_positif = np.argmax(proba_sample)
idx_negatif = np.argmin(proba_sample)

print(f"\n=== KASUS PREDIKSI NO-SHOW TERTINGGI (probabilitas={proba_sample[idx_positif]:.4f}) ===")
print(X_sample.iloc[idx_positif])
print(f"\n=== KASUS PREDIKSI NO-SHOW TERENDAH (probabilitas={proba_sample[idx_negatif]:.4f}) ===")
print(X_sample.iloc[idx_negatif])

plt.figure()
shap.force_plot(
    explainer.expected_value, shap_values[idx_positif], X_sample.iloc[idx_positif],
    feature_names=fitur, matplotlib=True, show=False
)
plt.savefig("shap_force_plot_positif_f2.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
shap.force_plot(
    explainer.expected_value, shap_values[idx_negatif], X_sample.iloc[idx_negatif],
    feature_names=fitur, matplotlib=True, show=False
)
plt.savefig("shap_force_plot_negatif_f2.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "#" * 70)
print("# SELESAI - REVISI BATCH 3 s.d. BATCH 6 (F2-SCORE)")
print("#" * 70)
print("\nLANGKAH SELANJUTNYA:")
print("1. Cek hasil Batch 5 -- konfirmasi model mana yang tetap unggul di Recall/F2")
print("2. Update file model di FastAPI: ganti model_xgb_final.pkl dengan model_xgb_final_f2.pkl")
print("3. Update THRESHOLD_SEDANG & THRESHOLD_TINGGI di .env sesuai hasil analisis di atas")
print("4. Jalankan ulang skrip cari_sampel_risiko_tinggi.py untuk data uji coba baru")
