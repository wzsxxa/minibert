import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset , DataLoader
from imageclsmodel import imageclsmodel
icm = imageclsmodel()


icm = torch.load('model50.pth')
test_data = datasets.FashionMNIST(root="data", train=False, download=False, transform=ToTensor())
cnt = 0
len = len(test_data)
print(len)
for i in range(len):
    sm = nn.Softmax(dim = 1)
    py = icm(test_data[i][0])
    py = sm(py)
    yhat =  py.argmax(1)
    print(f"{yhat}      {test_data[i][1]}")
    if yhat == test_data[i][1]:
        cnt = cnt + 1

print(f"accuracy:{cnt / len}")
# testloader = DataLoader(test_data , batch_size = 64  ,shuffle = True)