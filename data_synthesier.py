import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from nemo_microservices.data_designer.essentials import (
    DataDesignerConfigBuilder,
    InferenceParameters,
    ModelConfig,
    NeMoDataDesignerClient,
    SamplerColumnConfig,
    SamplerType,
    CategorySamplerParams,
)

# =====================================
# CONFIG
# =====================================
TARGET_ROWS = 10000
BATCH_SIZE = 100   # NIM limit
MAX_WORKERS = 5    # parallel threads

# =====================================
# CONNECT TO NIM
# =====================================
client = NeMoDataDesignerClient(
    base_url="https://ai.api.nvidia.com/v1/nemo/dd",
    default_headers={
        "Authorization": ""
    }
)

model_configs = [
    ModelConfig(
        alias="industrial-model",
        model="openai/gpt-oss-20b",
        inference_parameters=InferenceParameters(
            temperature=0.5,
            top_p=0.9,
            max_tokens=300,
        ),
    )
]

builder = DataDesignerConfigBuilder(model_configs)

# =====================================
# CATEGORICAL CONFIG
# =====================================
builder.add_column(
    SamplerColumnConfig(
        name="equipment",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Compressor", "Turbine", "Pump"]
        )
    )
)

builder.add_column(
    SamplerColumnConfig(
        name="location",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Atlanta", "Chicago", "Houston", "New York", "San Francisco"]
        )
    )
)

# =====================================
# PARALLEL NIM CALL FUNCTION
# =====================================
def fetch_batch():
    preview = client.preview(builder, num_records=BATCH_SIZE)
    return preview.dataset.copy()

# =====================================
# PARALLEL DATA GENERATION
# =====================================
print("🚀 Generating categorical data (parallel)...")

batches = []
num_batches = TARGET_ROWS // BATCH_SIZE

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(fetch_batch) for _ in range(num_batches)]

    for i, future in enumerate(as_completed(futures)):
        batch = future.result()
        batches.append(batch)
        print(f"Batch {i+1}/{num_batches} collected")

df = pd.concat(batches, ignore_index=True).iloc[:TARGET_ROWS]

print("✅ NIM categorical generation complete")

# =====================================
# NUMERICAL GENERATION (MATCH STATS)
# =====================================
stats = {
    "temperature": {"mean": 70.922478, "std": 16.200059, "min": 10.269385, "max": 149.690420},
    "pressure": {"mean": 35.738048, "std": 10.381593, "min": 3.620798, "max": 79.887734},
    "vibration": {"mean": 1.611809, "std": 0.728560, "min": -0.428188, "max": 4.990537},
    "humidity": {"mean": 55.0, "std": 15.0, "min": 20, "max": 90},
}

def generate_column(stat, size):
    data = np.random.normal(stat["mean"], stat["std"], size)
    data = (data - data.mean()) / data.std()
    data = data * stat["std"] + stat["mean"]
    return np.clip(data, stat["min"], stat["max"])

for col, stat in stats.items():
    df[col] = generate_column(stat, TARGET_ROWS)

# =====================================
# EXACT DISTRIBUTIONS
# =====================================
def scale_distribution(orig_dist, new_total):
    scale = new_total / sum(orig_dist.values())
    scaled = {k: int(v * scale) for k, v in orig_dist.items()}

    diff = new_total - sum(scaled.values())
    keys = list(scaled.keys())
    for i in range(abs(diff)):
        scaled[keys[i % len(keys)]] += 1 if diff > 0 else -1

    return scaled

equipment_orig = {"Compressor": 2573, "Turbine": 2565, "Pump": 2534}
location_orig = {"Atlanta": 1564, "Chicago": 1553, "Houston": 1548, "New York": 1526, "San Francisco": 1481}
faulty_orig = {0: 6905, 1: 767}

equipment_dist = scale_distribution(equipment_orig, TARGET_ROWS)
location_dist = scale_distribution(location_orig, TARGET_ROWS)
faulty_dist = scale_distribution(faulty_orig, TARGET_ROWS)

def create_exact_distribution(dist):
    values = []
    for k, v in dist.items():
        values.extend([k]*v)
    np.random.shuffle(values)
    return values

df["equipment"] = create_exact_distribution(equipment_dist)
df["location"] = create_exact_distribution(location_dist)

# =====================================
# FAULT GENERATION
# =====================================
score = (
    df["temperature"] * 0.3 +
    df["vibration"] * 0.5 +
    df["pressure"] * 0.2
)

threshold = np.percentile(score, 90)
df["faulty"] = (score > threshold).astype(int)

# enforce exact count
df["faulty"] = create_exact_distribution(faulty_dist)

# =====================================
# SAVE
# =====================================
df.to_csv("synthetic_nim_parallel_10000.csv", index=False)

print("\n✅ DONE: synthetic_nim_parallel_10000.csv")
print("Shape:", df.shape)