import torch
import numpy as np

data = [[1, 2], [3, 4]]

x_data = torch.tensor(data, dtype=torch.float32)

print(f"x_data: {x_data}")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np: {x_np}")

x_ones = torch.ones_like(x_data)
print(f"x_ones: {x_ones}")