import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad = True)
print(w)

b = torch.randn(3, requires_grad = True)
z = torch.matmul(x, w) + b
print(z)
#z1 = torch.matmul(w,x)+b
#print(z1)

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(z.grad_fn)
print(loss.grad_fn)

loss.backward()
print(w.grad)
print(b.grad)

print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

inp = torch.eye(4, 5, requires_grad = True)
print(inp)
out = (inp+1).pow(2).t()
print(out)

print(1234567)
out.backward(torch.ones_like(out), retain_graph=True)
print(inp.grad)
out.backward(torch.ones_like(out), retain_graph=True)
print(inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(inp.grad)

