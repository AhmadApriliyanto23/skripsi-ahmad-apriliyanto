"""
=====================================================================
BATCH 4 (REVISI) : MODEL FINAL (LANGSUNG DARI best_estimator_)
=====================================================================
Catatan: pada revisi F1 sebelumnya, Batch 4 melatih ULANG model dengan
parameter yang di-hardcode manual dari hasil Batch 3. Pada revisi ini,
model_rf dan model_xgb (best_estimator_ dari batch3_tuning_f2.py) sudah
otomatis dilatih pada seluruh X_train oleh GridSearchCV, sehingga TIDAK
PERLU dilatih ulang secara terpisah -- best_estimator_ dari GridSearchCV
sudah merupakan model final yang siap dipakai.

PRASYARAT: jalankan batch3_tuning_f2.py terlebih dahulu (butuh
model_rf_final_f2.pkl dan model_xgb_final_f2.pkl).

OUTPUT FILE (dipakai oleh batch5/threshold/batch6):
- X_test_f2.csv
- y_test_f2.csv
=====================================================================
"""

import joblib
import pandas as pd

from utils_fitur import fitur

print("\n" + "#" * 70)
print("# BATCH 4 (REVISI) : MODEL FINAL (LANGSUNG DARI best_estimator_)")
print("#" * 70)

model_rf = joblib.load("model_rf_final_f2.pkl")
model_xgb = joblib.load("model_xgb_final_f2.pkl")

df_test = pd.read_csv("data_test.csv", parse_dates=['tgl_pendaftaran', 'tgl_jadwal'])
X_test = df_test[fitur]
y_test = df_test['target']
X_test.to_csv("X_test_f2.csv", index=False)
y_test.to_csv("y_test_f2.csv", index=False)

print("\n=== FEATURE IMPORTANCE - RANDOM FOREST (F2) ===")
importance_rf = pd.DataFrame({
    'fitur': fitur, 'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_rf.to_string(index=False))

print("\n=== FEATURE IMPORTANCE - XGBOOST (F2) ===")
importance_xgb = pd.DataFrame({
    'fitur': fitur, 'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_xgb.to_string(index=False))

print("\n>>> LANJUT KE: batch5_evaluasi_f2.py <<<")
