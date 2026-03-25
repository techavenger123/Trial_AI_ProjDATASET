import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def column_probability_with_plots(df, bins=5):
    result = {}

    for col in df.columns:
        data = df[col].dropna()

        print(f"\nProcessing column: {col}")

        # Check if numeric
        if np.issubdtype(data.dtype, np.number):
            # If too many unique values → treat as continuous
            if data.nunique() > 20:
                print("→ Continuous data detected, applying binning")
                data = pd.cut(data, bins=bins)

        # Compute probabilities
        prob = data.value_counts(normalize=True).sort_index()
        result[col] = prob

        print(prob)

        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(x=prob.index.astype(str), y=prob.values)

        plt.title(f"Probability Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Probability")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return result
dataset = pd.read_csv('industrial-equipment-monitoring-dataset/versions/1/equipment_anomaly_data.csv')

# Drop unnecessary columns properly
dataset = dataset.drop(columns=['equipment', 'location'])

probabilities = column_probability_with_plots(dataset, bins=6)

# Only numeric columns
columns = dataset.select_dtypes(include=[np.number]).columns

for column in columns:
    data = dataset[column].dropna()

    # KDE Plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data, fill=True, color='skyblue')
    plt.title('Kernel Density Estimation of ' + column)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()

    # Fit normal distribution
    mu, sigma = stats.norm.fit(data)
    print(f"{column} → mean: {mu}, std: {sigma}")

    if sigma == 0:
        print(f"Skipping {column} (zero variance)\n")
        continue

    # PDF
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = stats.norm.pdf(x, loc=mu, scale=sigma)

    # Plot PDF + Histogram
    plt.figure(figsize=(8, 6))
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Normal PDF')

    sns.histplot(data, bins=100, stat="density",
                 color='gray', alpha=0.5, label='Data Histogram')

    plt.title('Fitted Normal Distribution to Data: ' + column)
    plt.xlabel(column)
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()