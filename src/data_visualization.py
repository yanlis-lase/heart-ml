import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Konfigurasi awal
# ===============================
DATA_PATH = "data/heart_clean.xlsx"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ===============================
# 2. Load dataset
# ===============================
df = pd.read_excel(DATA_PATH)

print("Dataset loaded:", df.shape)

# ===============================
# 3. Histogram beberapa fitur
# ===============================
hist_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

for col in hist_columns:
    plt.figure(figsize=(7, 5))
    sns.histplot(df[col], bins=20)
    plt.title(f"Histogram {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    save_path = f"{OUTPUT_DIR}/hist_{col}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)

# ===============================
# 4. Boxplot untuk deteksi outlier
# ===============================
boxplot_columns = ["trestbps", "chol", "thalach", "oldpeak"]

for col in boxplot_columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot {col}")

    save_path = f"{OUTPUT_DIR}/boxplot_{col}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)

# ===============================
# 5. Correlation Heatmap
# ===============================
plt.figure(figsize=(12, 10))
corr = df.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Feature Correlation Heatmap")

heatmap_path = f"{OUTPUT_DIR}/correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", heatmap_path)

print("\nVisualization generation completed.")