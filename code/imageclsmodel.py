#步骤
# 1. 下载数据集(dataset)
# 2. 使用dataloader
# 3. 定义模型
# 4. 训练模型

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset , DataLoader



class imageclsmodel(nn.Module):
    def __init__(self):
        super(imageclsmodel, self).__init__()
        self.flatten = nn.Flatten()
        self.lrs = nn.Sequential(
            nn.Linear(28 * 28 , 512),
            nn.ReLU(),
            nn.Linear(512 , 512),
            nn.ReLU(),
            nn.Linear(512 , 10),
        )

    def forward(self , x):
        x = self.flatten(x)
        res = self.lrs(x)
        return res


if __name__ == '__main__':
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    training_data = datasets.FashionMNIST(root = "data" , train = True , download= False
                                          ,  transform=ToTensor())
    # print(training_data[10000][1])
    test_data = datasets.FashionMNIST(root="data", train=False, download=True , transform=ToTensor())
    trainloader = DataLoader(  training_data,batch_size = 256 , shuffle = True , )
    testloader = DataLoader(test_data , batch_size = 64  ,shuffle = True)
    icm = imageclsmodel()
    icm = icm.to(device)
    epoch = 50
    lr = 1e-3
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(icm.parameters() , lr = lr)
    for i in range(epoch):
        print(f"epoch {i + 1}")
        for batch , (x , y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            py = icm(x)
            loss = lf(py , y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if batch % 100 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f}")



    torch.save(icm , 'model50.pth')


