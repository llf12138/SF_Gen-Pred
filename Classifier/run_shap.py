import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# 1. Load the XGBoost model
print("Loading XGBoost model...")
model = joblib.load("xgb_classifier_model.pkl")
print("Model loaded.")

# 2. Load test features and sample for SHAP
X_test = pd.read_csv("X_test_for_shap.csv")
X_shap = X_test.sample(n=5000, random_state=42)
print(f"SHAP sample size: {len(X_shap)}")

# 3. Compute SHAP values (TreeExplainer for XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# Binary classification: keep SHAP values of the positive class
sv_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
print(f"SHAP shape: {sv_pos.shape}")

# 4. Save SHAP values
shap_df = pd.DataFrame(sv_pos, columns=X_shap.columns)
shap_df.to_csv("shap_values_xgb.csv", index=False, encoding="utf-8-sig")

# 5. Plots
plt.rcParams.update({
    "font.size": 11,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.dpi": 300,
    "figure.figsize": (10, 8)
})

# Bar plot of top-10 feature importance
plt.figure()
shap.summary_plot(sv_pos, X_shap, max_display=10, plot_type="bar", show=False)
plt.title("Top 10 Feature Importance (XGBoost)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("shap_top10_bar_xgb.png", bbox_inches="tight")
plt.close()

# Dot (beeswarm) plot
plt.figure()
shap.summary_plot(sv_pos, X_shap, max_display=10, show=False)
plt.title("Top 10 SHAP Dot Plot (XGBoost)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("shap_top10_dot_xgb.png", bbox_inches="tight")
plt.close()

print("SHAP analysis complete.")
