import pandas as pd

# Load CSV
df = pd.read_csv("synthetic_nim_parallel_10000.csv")

# -----------------------------
# 1. Basic Info
# -----------------------------
print("\n=== BASIC INFO ===")
print(df.info())

# -----------------------------
# 2. Descriptive Statistics
# -----------------------------
print("\n=== NUMERICAL STATS ===")
print(df.describe())

print("\n=== ALL COLUMNS STATS ===")
print(df.describe(include='all'))

# -----------------------------
# 3. Missing Values
# -----------------------------
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# -----------------------------
# 4. Unique Values Count
# -----------------------------
print("\n=== UNIQUE VALUE COUNT ===")
print(df.nunique())

# -----------------------------
# 5. Frequency of Each Unique Value (ALL COLUMNS)
# -----------------------------
print("\n=== VALUE FREQUENCY PER COLUMN ===")

for col in df.columns:
    print(f"\n--- Column: {col} ---")
    print(df[col].value_counts(dropna=False))

# -----------------------------
# 6. Correlation (Numerical Only)
# -----------------------------
print("\n=== CORRELATION MATRIX ===")
print(df.corr(numeric_only=True))

with open("analysis/summary_report.txt", "w") as f:
    f.write("=== BASIC INFO ===\n")
    df.info(buf=f)

    f.write("\n\n=== DESCRIBE ===\n")
    f.write(str(df.describe(include='all')))

    f.write("\n\n=== MISSING VALUES ===\n")
    f.write(str(df.isnull().sum()))

    f.write("\n\n=== UNIQUE COUNTS ===\n")
    f.write(str(df.nunique()))

    f.write("\n\n=== VALUE COUNTS ===\n")
    for col in df.columns:
        f.write(f"\n--- {col} ---\n")
        f.write(str(df[col].value_counts(dropna=False)))

summary = df.agg(['count', 'nunique'])
print(summary)

from ydata_profiling import ProfileReport

profile = ProfileReport(df, explorative=True)
profile.to_file("report.html")