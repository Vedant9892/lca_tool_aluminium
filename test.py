import joblib, pandas as pd

# Load bundle
bundle = joblib.load("models/trained/energy_predictor.pkl")
model, pre = bundle["model"], bundle["pre"]

# Load a batch of feature rows (drop targets if present)
df = pd.read_csv("data/processed/train_1.csv")
X = df.drop(columns=[c for c in ["energy_kwh_per_kg","co2_kg_per_kg","production_cost_usd_per_kg"] if c in df.columns])

# Transform and predict
Xp = pre.transform(X)
y_pred = model.predict(Xp)

print(y_pred[:10])
