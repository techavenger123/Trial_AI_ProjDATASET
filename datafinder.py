import pandas as pd

data=pd.read_csv("synthetics3/synthetic/dense_results.csv")

columns=data.columns

print(columns)


import pandas as pd

df = data

cols = ["n_estimators", "max_features", "split_label",
        "val_auc", "val_f1", "val_recall", "val_precision",
        "test_f1", "test_auc", "overfit_auc"]

print("=" * 60)
print("TOP 5 BY VAL F1")
print(df.sort_values("val_f1", ascending=False).head(5)[cols].to_string(index=False))

print("\n" + "=" * 60)
print("TOP 5 BY VAL F1 WITH LOW OVERFIT (overfit_auc < 0.05)")
low_overfit = df[df["overfit_auc"] < 0.05]
print(low_overfit.sort_values("val_f1", ascending=False).head(5)[cols].to_string(index=False))

print("\n" + "=" * 60)
print("TOP 5 BY VAL RECALL (fault detection priority)")
print(df.sort_values(["val_recall", "val_f1"], ascending=False).head(5)[cols].to_string(index=False))

print("\n" + "=" * 60)
print("OVERALL STATS")
print(df[["val_f1", "val_auc", "val_recall", "overfit_auc"]].describe().round(4))