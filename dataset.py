import kagglehub

# Download latest version
path = kagglehub.dataset_download("dnkumars/industrial-equipment-monitoring-dataset")

print("Path to dataset files:", path)