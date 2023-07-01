import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

temp = np.array(data)
x_np = torch.from_numpy(temp)
print(x_np)

x_ones = torch.ones_like(x_data)
print(x_ones)
x_rand = torch.rand_like(x_data, dtype = torch.float)
print(x_rand)

shape1 = (2,3,)
x1 = torch.ones(shape1)
shape2 = (3,1,)
x2 = torch.zeros(shape2)
shape3 = (1,4,)
x3 = torch.rand(shape3)

print(x1)
print(x2)
print(x3)

sample_tensor = torch.rand(3,2)
print(sample_tensor.shape)
print(sample_tensor.dtype)
print(sample_tensor.device)

tensor = torch.ones(4,4)
if torch.cuda.is_available():
    sample_tensor = sample_tensor.to("cuda")
    tensor = tensor.to("cuda")

tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:,1] = 3
print(tensor)

y1 = tensor @ tensor.T
print(y1)
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)

z1 = tensor*tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(y1)
torch.mul(tensor, tensor, out = z3)
print(z3)

tensor.add_(4)
print(tensor)

t = torch.ones(5)
print(t)
n = t.numpy()
print(n)
t.add_(2)
print(n)

n0 = np.ones(3)
t0 = torch.from_numpy(n0)
np.add(n0, 1, out = n0)
print(t0)
