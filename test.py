import torch
from torch.autograd import Variable
def f(x):
    y = x ** 3
    return y

x = Variable(torch.tensor([5.0]), requires_grad=True)
loss = f(x)
grad_x = torch.autograd.grad(loss, x, create_graph=True)
print(grad_x)  # 一阶导数 75
print(torch.norm(grad_x[0], p=2))
loss = loss+torch.norm(grad_x[0], p=2)**2
loss.backward()
print(x.grad) # buffer内存的是30，也就是二阶导数的值