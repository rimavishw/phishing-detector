import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from features import extract_features

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("XGBoost not available, skipping it.")

df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"])
df = df.drop_duplicates(subset="URL")
df = df.sample(min(len(df), 60000), random_state=42)   # keeps training fast

y = (df["label"] == 0).astype(int)    # 1 = phishing (verify this against your data!)
X = pd.DataFrame([extract_features(u) for u in df["URL"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6,
                                      learning_rate=0.1, eval_metric="logloss")

best_name, best_f1, best_model = None, 0, None
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred)
    print(f"\n===== {name} | F1 = {f1:.4f} =====")
    print(classification_report(y_test, pred, target_names=["legit", "phishing"]))
    print(confusion_matrix(y_test, pred))
    if f1 > best_f1:
        best_name, best_f1, best_model = name, f1, model

print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")
joblib.dump({"model": best_model, "columns": list(X.columns)}, "model/phishing_model.joblib")