import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
# 1. Load
df = pd.read_csv('path to your training dataset')
X = df.drop(['App', 'Label'], axis=1)
y = df['Label']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)

# 5. XGBoost + GridSearch
param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ),
    param_grid,
    scoring='f1',
    cv=cv,
    verbose=2,
    n_jobs=-1
)
grid.fit(X_resampled, y_resampled)
best_model = grid.best_estimator_

# 6. Evaluate
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Threshold tuning
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
best_f1, best_thr = 0, 0.5
for thr in [i / 100 for i in range(25, 76)]:
    preds = (y_proba >= thr).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1, best_thr = score, thr
print(f'Best F1 {best_f1:.3f} at threshold {best_thr}')

# compatability issues
# joblib.dump(best_model, 'detect_malware_x.pkl')

joblib.dump(scaler, 'scaler_x_new.pkl')

# Save model in version-safe JSON format 
best_model.get_booster().save_model('detect_malware_x_new.pkl')

print("✅ Model & Scaler saved successfully")



