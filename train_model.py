"""
Creates wine_data.csv and best_wine_model.pkl.
Skip this if your notebook already produced them (just copy them next to app.py).

    python train_model.py
"""
import joblib
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_wine(as_frame=True)
df = data.frame
df.to_csv("wine_data.csv", index=False)

X, y = df.drop(columns="target"), df["target"]
candidates = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
}
scores = {n: cross_val_score(m, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean() for n, m in candidates.items()}
best = max(scores, key=scores.get)
print(scores, "->", best)
candidates[best].fit(X, y)
joblib.dump(candidates[best], "best_wine_model.pkl")
