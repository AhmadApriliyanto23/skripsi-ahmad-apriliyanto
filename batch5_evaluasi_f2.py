"""
=====================================================================
BATCH 5 (REVISI) : EVALUASI & PERBANDINGAN MODEL (HASIL TUNING F2)
=====================================================================
PRASYARAT: jalankan batch3_tuning_f2.py dan batch4_feature_importance_f2.py
terlebih dahulu (butuh model_rf_final_f2.pkl, model_xgb_final_f2.pkl,
X_test_f2.csv, y_test_f2.csv).

OUTPUT FILE:
- hasil_evaluasi_model_f2.csv
- confusion_matrix_comparison_f2.png
- roc_curve_comparison_f2.png
=====================================================================
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

print("\n" + "#" * 70)
print("# BATCH 5 (REVISI) : EVALUASI & PERBANDINGAN MODEL (HASIL TUNING F2)")
print("#" * 70)

model_rf = joblib.load("model_rf_final_f2.pkl")
model_xgb = joblib.load("model_xgb_final_f2.pkl")
X_test = pd.read_csv("X_test_f2.csv")
y_test = pd.read_csv("y_test_f2.csv").squeeze("columns")

y_pred_rf = model_rf.predict(X_test)
y_proba_rf = model_rf.predict_proba(X_test)[:, 1]
y_pred_xgb = model_xgb.predict(X_test)
y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]


def hitung_metrik(y_true, y_pred, y_proba, nama_model):
    return {
        'Model': nama_model,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'F2-Score': fbeta_score(y_true, y_pred, beta=2),
        'AUC-ROC': roc_auc_score(y_true, y_proba)
    }


hasil_rf = hitung_metrik(y_test, y_pred_rf, y_proba_rf, 'Random Forest')
hasil_xgb = hitung_metrik(y_test, y_pred_xgb, y_proba_xgb, 'XGBoost')

df_hasil = pd.DataFrame([hasil_rf, hasil_xgb])
print("\n=== TABEL PERBANDINGAN METRIK EVALUASI (HASIL TUNING F2) ===")
print(df_hasil.to_string(index=False))
df_hasil.to_csv("hasil_evaluasi_model_f2.csv", index=False)

print("\n=== CLASSIFICATION REPORT - RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=['HADIR', 'TIDAK HADIR']))
print("\n=== CLASSIFICATION REPORT - XGBOOST ===")
print(classification_report(y_test, y_pred_xgb, target_names=['HADIR', 'TIDAK HADIR']))

cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\n=== CONFUSION MATRIX - RANDOM FOREST ===")
print(pd.DataFrame(cm_rf, index=['Aktual: HADIR', 'Aktual: TIDAK HADIR'],
                    columns=['Prediksi: HADIR', 'Prediksi: TIDAK HADIR']))
print("\n=== CONFUSION MATRIX - XGBOOST ===")
print(pd.DataFrame(cm_xgb, index=['Aktual: HADIR', 'Aktual: TIDAK HADIR'],
                    columns=['Prediksi: HADIR', 'Prediksi: TIDAK HADIR']))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, judul in zip(axes, [cm_rf, cm_xgb], ['Random Forest', 'XGBoost']):
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f'Confusion Matrix - {judul} (F2-tuned)')
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['HADIR', 'TIDAK HADIR'])
    ax.set_yticklabels(['HADIR', 'TIDAK HADIR'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix_comparison_f2.png", dpi=150)
plt.close()

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
plt.figure(figsize=(7, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {hasil_rf['AUC-ROC']:.4f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {hasil_xgb['AUC-ROC']:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Kurva ROC - Perbandingan Model (F2-tuned)'); plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison_f2.png", dpi=150)
plt.close()

print("\n=== RINGKASAN PERBANDINGAN ANTAR METRIK (HASIL TUNING F2) ===")
for metrik in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score', 'AUC-ROC']:
    nilai_rf = df_hasil.loc[df_hasil['Model'] == 'Random Forest', metrik].values[0]
    nilai_xgb = df_hasil.loc[df_hasil['Model'] == 'XGBoost', metrik].values[0]
    unggul = 'Random Forest' if nilai_rf > nilai_xgb else ('XGBoost' if nilai_xgb > nilai_rf else 'Sebanding')
    print(f"{metrik:12s} -> RF: {nilai_rf:.4f} | XGBoost: {nilai_xgb:.4f} | Lebih unggul: {unggul}")

print("\n*** PERIKSA HASIL DI ATAS: apakah XGBoost tetap unggul di Recall/F2? ***")
print("*** Jika ya, kesimpulan pemilihan model tetap konsisten dengan sebelumnya ***")
print("*** Jika tidak, kesimpulan model terpilih PERLU ditinjau ulang ***")

print("\n>>> LANJUT KE: analisis_threshold_f2.py (lalu batch6_shap_f2.py) <<<")
