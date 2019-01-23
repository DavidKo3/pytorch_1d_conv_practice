import torch
import src.mnistModel as model

net = model.MNISTConvNet()
print(net)


input = torch.randn(1, 1, 28, 28)

print(input.shape)

out = net(input)
print(out.size())