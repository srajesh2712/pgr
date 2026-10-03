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


if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator().type)
    print(f"Device tensor is stored on: {tensor.device}")   


tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(f"Modified tensor: {tensor}")


t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Concatenated tensor: {t1}")

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(f"Matrix multiplication 1: {y1}")
print(f"Matrix multiplication 2: {y2}") 
print(f"Matrix multiplication 3: {y3}")

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(f"Element-wise multiplication 1: {z1}")
print(f"Element-wise multiplication 2: {z2}")
print(f"Element-wise multiplication 3: {z3}")

agg = tensor.sum()
agg_item = agg.item()
print(f"Sum of tensor: {agg}")
print(f"Sum of tensor as item: {agg_item}") 