"""
=====================================================================
BATCH 3 : CLASS IMBALANCE + TUNING (F2-SCORE)
=====================================================================
Latar belakang revisi:
Pada penelitian tahap awal, GridSearchCV menggunakan scoring='f1' untuk
mencari parameter terbaik, namun model final dipilih berdasarkan Recall
(prioritas meminimalkan false negative). Hal ini menimbulkan inkonsistensi
antara kriteria tuning dan tujuan akhir sistem.

Revisi ini mengganti scoring GridSearchCV menjadi F2-score (fbeta_score
dengan beta=2), yang memberi bobot recall 2x lebih besar dibanding
precision -- selaras dengan prioritas meminimalkan false negative, namun
tetap mempertimbangkan precision sebagai penyeimbang (tidak sepenuhnya
condong ke recall murni seperti scoring='recall').

CATATAN: Batch 1 dan Batch 2 TIDAK diulang -- data_train.csv dan
data_test.csv hasil sebelumnya tetap dipakai, karena revisi ini hanya
menyangkut kriteria pemilihan parameter model, bukan pra pemrosesan
atau pembagian data.

OUTPUT FILE (dipakai oleh batch 4/5/6):
- model_rf_final_f2.pkl
- model_xgb_final_f2.pkl
- cv_results_rf_f2.csv
- cv_results_xgb_f2.csv
=====================================================================
"""

import time
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score, make_scorer

from utils_fitur import fitur

# F2-score scorer: beta=2 memberi bobot recall 2x lebih besar dari precision
f2_scorer = make_scorer(fbeta_score, beta=2)

print("\n" + "#" * 70)
print("# BATCH 3 (REVISI) : CLASS IMBALANCE + TUNING (F2-SCORE)")
print("#" * 70)

df_train = pd.read_csv("data_train.csv", parse_dates=['tgl_pendaftaran', 'tgl_jadwal'])
X_train = df_train[fitur]
y_train = df_train['target']

jumlah_kelas = y_train.value_counts()
rasio_imbalance = jumlah_kelas[0] / jumlah_kelas[1]
scale_pos_weight_value = jumlah_kelas[0] / jumlah_kelas[1]
print(f"Rasio kelas mayoritas : minoritas = {rasio_imbalance:.2f} : 1")
print(f"scale_pos_weight (acuan XGBoost): {scale_pos_weight_value:.4f}")

tscv = TimeSeriesSplit(n_splits=5)

# --- GridSearchCV: Random Forest (scoring=F2, class_weight tetap 'balanced') ---
print("\n" + "=" * 60)
print("TUNING RANDOM FOREST (F2-SCORE)")
print("=" * 60)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42, n_jobs=1)

grid_rf = GridSearchCV(
    estimator=rf, param_grid=param_grid_rf, cv=tscv,
    scoring=f2_scorer, n_jobs=2, verbose=2, error_score='raise'
)

start_time = time.time()
grid_rf.fit(X_train, y_train)
print(f"\nWaktu tuning Random Forest: {time.time() - start_time:.2f} detik")
print(f"Parameter terbaik RF: {grid_rf.best_params_}")
print(f"Skor F2 CV terbaik RF: {grid_rf.best_score_:.4f}")

pd.DataFrame(grid_rf.cv_results_).to_csv("cv_results_rf_f2.csv", index=False)

# --- GridSearchCV: XGBoost (scoring=F2) ---
print("\n" + "=" * 60)
print("TUNING XGBOOST (F2-SCORE)")
print("=" * 60)

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'scale_pos_weight': [1, scale_pos_weight_value]
}

xgb = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1)

grid_xgb = GridSearchCV(
    estimator=xgb, param_grid=param_grid_xgb, cv=tscv,
    scoring=f2_scorer, n_jobs=2, verbose=2, error_score='raise'
)

start_time = time.time()
grid_xgb.fit(X_train, y_train)
print(f"\nWaktu tuning XGBoost: {time.time() - start_time:.2f} detik")
print(f"Parameter terbaik XGBoost: {grid_xgb.best_params_}")
print(f"Skor F2 CV terbaik XGBoost: {grid_xgb.best_score_:.4f}")

pd.DataFrame(grid_xgb.cv_results_).to_csv("cv_results_xgb_f2.csv", index=False)

print("\n=== RINGKASAN HASIL TUNING (F2-SCORE) ===")
print(f"Random Forest - Best F2 (CV): {grid_rf.best_score_:.4f} | Params: {grid_rf.best_params_}")
print(f"XGBoost - Best F2 (CV): {grid_xgb.best_score_:.4f} | Params: {grid_xgb.best_params_}")

# PENTING: model final langsung diambil dari best_estimator_ GridSearchCV,
# BUKAN ditulis ulang manual -- karena parameter terbaik belum diketahui
# sebelum kode ini dijalankan (berbeda dengan hasil tuning F1 sebelumnya).
# best_estimator_ SUDAH otomatis dilatih ulang pada seluruh X_train oleh
# GridSearchCV (refit=True secara default), jadi model ini siap dipakai
# langsung sebagai model final -- tidak perlu training ulang terpisah
# (lihat catatan di batch4_feature_importance_f2.py).
model_rf = grid_rf.best_estimator_
model_xgb = grid_xgb.best_estimator_

joblib.dump(model_rf, "model_rf_final_f2.pkl")
joblib.dump(model_xgb, "model_xgb_final_f2.pkl")
print("\nModel final (hasil tuning F2) tersimpan:")
print("model_rf_final_f2.pkl, model_xgb_final_f2.pkl")

print("\n>>> LANJUT KE: batch4_feature_importance_f2.py <<<")
