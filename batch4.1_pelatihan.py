import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import pandas as pd

from utils_fitur import fitur

df_train = pd.read_csv("data_train.csv", parse_dates=['tgl_pendaftaran', 'tgl_jadwal'])
X_train = df_train[fitur]
y_train = df_train['target']

# === Random Forest ===
rf_final = RandomForestClassifier(
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=4,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)

start = time.time()
rf_final.fit(X_train, y_train)
waktu_rf = time.time() - start
print(f"Waktu pelatihan RF: {waktu_rf:.2f} detik")

# === XGBoost ===
xgb_final = XGBClassifier(
    learning_rate=0.05,
    max_depth=7,
    n_estimators=100,
    scale_pos_weight=3.96725938470223,
    subsample=0.7,
    random_state=42
)

start = time.time()
xgb_final.fit(X_train, y_train)
waktu_xgb = time.time() - start
print(f"Waktu pelatihan XGBoost: {waktu_xgb:.2f} detik")