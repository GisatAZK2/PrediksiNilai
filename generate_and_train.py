import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle

# ======= Generate Synthetic Data =======
np.random.seed(42)
n = 10000

data = pd.DataFrame({
    'mata_pelajaran': np.random.choice(['Matematika', 'IPA', 'IPS', 'Bahasa', 'Seni', 'Agama'], n),
    'jam_belajar': np.clip(np.random.normal(5, 2, n), 0, 10),
    'nilai_ujian_sebelumnya': np.clip(np.random.normal(70, 15, n), 0, 100),  # 0-100 supaya ada nilai rendah juga
    'jam_tidur': np.clip(np.random.normal(7, 1.5, n), 4, 10),
    'tingkat_ekonomi': np.random.randint(1, 6, n),
    'tingkat_motivasi': np.random.randint(1, 6, n),
    'pertemanan': np.random.randint(1, 6, n),
})

# Flag nilai ujian sebelumnya rendah (≤30)
data['nilai_ujian_rendah_flag'] = (data['nilai_ujian_sebelumnya'] <= 30).astype(int)

# ======= Target Function =======
def generate_target(row):
    if row['nilai_ujian_sebelumnya'] <= 30:
        # Nilai rendah dipaksa jadi rendah (misal 0-30)
        return np.clip(np.random.normal(15, 7), 0, 30)
    else:
        # Target umum (nilai bagus tetap diperhitungkan normal)
        score = 1.0 * row['nilai_ujian_sebelumnya']
        score += 0.5 * row['jam_belajar']
        score += 1.0 * row['tingkat_motivasi']
        score += 0.5 * row['tingkat_ekonomi']
        score += 0.3 * row['pertemanan']
        score -= 2.0 * abs(row['jam_tidur'] - 7)
        noise = np.random.normal(0, 1.0)
        return np.clip(score + noise, 0, 100)

data['nilai_akhir'] = data.apply(generate_target, axis=1)
data.to_csv("data_sintetik_final_flag_v2.csv", index=False)

# ======= Split Dataset =======
X = data.drop(columns=['nilai_akhir', 'mata_pelajaran'])  # Hapus 'mata_pelajaran'
y = data['nilai_akhir']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======= Model Pipeline (Optional scaling numeric fitur) =======
# Kalau mau scaling bisa ditambahkan, tapi XGBoost ga wajib scaling
model = Pipeline([
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# ======= Hyperparameter Tuning =======
param_dist = {
    'regressor__n_estimators': [200, 300, 400],
    'regressor__max_depth': [3, 4, 5, 6],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__subsample': [0.9, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0],
    'regressor__reg_alpha': [0, 0.1, 0.5],
    'regressor__reg_lambda': [1, 1.5, 2]
}

search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# ======= Train =======
search.fit(X_train, y_train)
best_model = search.best_estimator_

# ======= Evaluasi =======
y_pred = best_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ======= Save model =======
with open('model_flag_v2.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("✅ Model sudah dilatih dan disimpan sebagai model_flag_v2.pkl")

# ======= Feature importance =======
importances = best_model.named_steps['regressor'].feature_importances_
feature_names = X_train.columns

print("\n🎯 Fitur Penting:")
for name, score in zip(feature_names, importances):
    print(f"{name}: {score:.4f}")
