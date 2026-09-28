import numpy as np
import pandas as pd
from pathlib import Path
import pyttb as ttb

df = pd.read_csv(Path.home() / "Capstone-Project" / "data" / "tensor_20x6x10.csv")

# drop id, keep feature columns in T0_F0 ... T5_F9 order
values = df.drop(columns=["sample_id"]).to_numpy(dtype=np.float32)

# (20 samples, 6 timesteps, 10 features)
arr = values.reshape(20, 6, 10)
tensor = ttb.tensor(arr)  # wrap as pyttb tensor for cp_als

print(arr.shape)  # (20, 6, 10)
print(arr[0])     # first sample: 6 x 10

np.random.seed(42)

Rank = 2
M1, M1_init, M1_info = ttb.cp_als(tensor, Rank)
print(M1)
print(M1_init)
print(M1_info)

print("-"*50)
# Hand-made 3 x 2 x 3 tensor
Rank2 = 3
tensor2 = [
    [[72, 80, 81], [1.7, 2.1, 2.2]],
    [[88, 90, 87], [2.5, 2.4, 2.3]],
    [[79, 75, 77], [1.2, 1.4, 1.5]]
]
arr2 = np.array(tensor2)  # (patient, feature, time)

# z-score each feature (axis 1) across patients and time
mean2 = arr2.mean(axis=(0, 2), keepdims=True)
std2 = arr2.std(axis=(0, 2), keepdims=True)
arr2_std = (arr2 - mean2) / std2
print(arr2_std)

tensor2 = ttb.tensor(arr2_std)  # cp_als needs a pyttb tensor

M2, M2_init, M2_info = ttb.cp_als(tensor2, Rank2)
print(M2)
print(M2_info)