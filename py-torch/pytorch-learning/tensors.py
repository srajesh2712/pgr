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

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"x_rand: {x_rand}")


shape = (2, 3,)
rand_tensor = torch.rand(shape)
one_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)
print(f"Random Tensor: {rand_tensor}")
print(f"Ones Tensor: {one_tensor}")
print(f"Zeros Tensor: {zero_tensor}")

tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Data type of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")