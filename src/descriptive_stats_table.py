import os
import sys
import pandas as pd

# OPTIONAL: untuk export tabel jadi gambar (PNG)
try:
    import dataframe_image as dfi
    HAS_DFI = True
except ImportError:
    HAS_DFI = False


def build_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"]).copy()

    # biasanya 'target' tidak masuk tabel statistik fitur
    if "target" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["target"])

    desc = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T

    desc = desc.rename(columns={
        "count": "Count",
        "mean": "Mean",
        "std": "Std",
        "min": "Min",
        "25%": "Q1 (25%)",
        "50%": "Median (50%)",
        "75%": "Q3 (75%)",
        "max": "Max"
    })

    desc = desc[["Count", "Mean", "Std", "Min", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max"]]

    desc["Count"] = desc["Count"].astype(int)
    numeric_cols = [c for c in desc.columns if c != "Count"]
    desc[numeric_cols] = desc[numeric_cols].round(3)

    desc.index.name = "Feature"
    desc = desc.reset_index()
    return desc


def main():
    # Pastikan path selalu relatif ke root project (folder saat script dijalankan)
    cwd = os.getcwd()
    print("=== Descriptive Stats Generator ===")
    print("Current working directory:", cwd)

    data_path = os.path.join("data", "heart_clean.xlsx")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("Looking for dataset at:", os.path.abspath(data_path))

    if not os.path.exists(data_path):
        print("\n[ERROR] Dataset tidak ditemukan.")
        print("Pastikan file ada di:", os.path.abspath(data_path))
        print("\nJika file kamu ada di root project (bukan di folder data), ubah DATA PATH jadi:")
        print('data_path = "heart_clean.xlsx"')
        sys.exit(1)

    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print("\n[ERROR] Gagal membaca file Excel:", e)
        sys.exit(1)

    print("Dataset loaded:", df.shape)
    stats_table = build_descriptive_stats(df)
    print("Table generated. Rows:", len(stats_table))

    xlsx_path = os.path.join(output_dir, "table_2_2_descriptive_stats.xlsx")
    csv_path = os.path.join(output_dir, "table_2_2_descriptive_stats.csv")

    stats_table.to_excel(xlsx_path, index=False)
    stats_table.to_csv(csv_path, index=False)

    print("\nSaved:")
    print("-", os.path.abspath(xlsx_path))
    print("-", os.path.abspath(csv_path))

    if HAS_DFI:
        png_path = os.path.join(output_dir, "table_2_2_descriptive_stats.png")
        try:
            dfi.export(stats_table, png_path)
            print("-", os.path.abspath(png_path))
        except Exception as e:
            print("\n[WARNING] dataframe-image terinstall, tapi export PNG gagal:", e)
            print("Excel/CSV tetap berhasil dibuat.")
    else:
        print("\n[Info] Jika ingin export tabel jadi PNG, install dulu:")
        print("pip install dataframe-image")

    print("\nDone.")


if __name__ == "__main__":
    main()
