import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

df = pd.read_csv("customer_data.csv")
pipe = joblib.load("best_churn_pipeline.pkl")

# Rebuild the exact features used in main.py
d = df.copy()
d["balance_per_product"] = d["balance"] / d["products_number"]
ratio = (d["estimated_salary"] / d["balance"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
d["salary_balance_ratio"] = ratio.fillna(ratio.median())
d["age_group"] = pd.cut(d["age"], [0, 25, 35, 45, 55, 65, 100],
                        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(object)
d["tenure_bucket"] = pd.cut(d["tenure"], [-1, 0, 2, 5, 10, 100],
                            labels=["0", "1-2", "3-5", "6-10", "10+"]).astype(object)
d["high_balance"] = (d["balance"] > d["balance"].quantile(0.75)).astype(int)
X = d.drop(columns=["customer_id", "churn"])

# Transform with the fitted preprocessor, keep readable names
pre = pipe.named_steps["preprocessor"]
clf = pipe.named_steps["classifier"]
X_t = pre.transform(X)
names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
X_t = pd.DataFrame(X_t, columns=names)

# SHAP
sample = X_t.sample(2000, random_state=42)
explainer = shap.TreeExplainer(clf)
sv = explainer(sample)

plt.figure()
shap.plots.bar(sv, max_display=15, show=False)
plt.tight_layout(); plt.savefig("shap_summary_bar.png", dpi=150); plt.close()

plt.figure()
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.tight_layout(); plt.savefig("shap_beeswarm.png", dpi=150); plt.close()

proba = clf.predict_proba(sample)[:, 1]
top = int(np.argmax(proba))
plt.figure()
shap.plots.waterfall(sv[top], max_display=12, show=False)
plt.tight_layout(); plt.savefig("shap_customer_waterfall.png", dpi=150); plt.close()

imp = pd.Series(np.abs(sv.values).mean(axis=0), index=names).sort_values(ascending=False)
print("Top churn drivers (mean |SHAP|):")
print(imp.head(10).round(4).to_string())