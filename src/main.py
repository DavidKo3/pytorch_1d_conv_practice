import torch
import src.mnistModel as model
import torch.nn as nn

net = model.MNISTConvNet()
print(net)


input = torch.randn(1, 28, 28)

# add 1 dim to 0 dim

out = net(input)
print("out.shape :", out.shape)


target = torch.tensor([3], dtype=torch.long)
print("target", target.shape)
loss_fn = nn.CrossEntropyLoss()
err = loss_fn(out, target)
err.backward()

print(err)
