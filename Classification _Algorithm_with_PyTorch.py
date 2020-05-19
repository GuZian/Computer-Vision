#Classification Algorithm with PyTorch
#by Zian Gu
#05/18/2020

#导入所需的包
import torch
import torchvision
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np

#print(torch.cuda.is_available());
local_file = "C:\\Users\\gu573\\Documents\\GitHub\\Computer Vision\\classification algorithm"
batch_size = 32  #批大小
learning_rate = 0.015  #学习率
momentum = 0.5  #用于随机梯度下降算法的参数，可以起到加速效果
num_epochs = 5  #迭代次数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#显示图片
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for index,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        prediction = model(data)
        loss = F.nll_loss(prediction,target)
        #SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print("Train epoch:{},interation:{},Loss:{}".format(epoch,index,loss.item()))
#测试
def test(model,device,test_loader):
    model.eval()
    total_loss=0.
    correct=0.
    with torch.no_grad():
        for index,(data,target) in enumerate(test_loader):
          data,target = data.to(device),target.to(device)
          output = model(data)
          total_loss += F.nll_loss(output,target,reduction="sum").item()
          prediction=output.argmax(dim=1)
          correct+=prediction.eq(target.view_as(prediction)).sum().item()

    total_loss/=len(test_loader.dataset)
    accuracy=correct/len(test_loader.dataset)*100.
    print("Test loss:{}，Accuracy:{}\n".format(total_loss,accuracy))

#神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(4 * 4 * 50,500)
        self.fc2 = nn.Linear(500,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)

#Data set
train_data = torchvision.datasets.MNIST(root=local_file, 
                                           train=True, 
                                           transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

test_data = torchvision.datasets.MNIST(root=local_file, 
                                          train=False, 
                                          transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          pin_memory=True)
                                          

#img_show(train_data[223][0].reshape(28,28))#显示图片
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)#随机梯度下降算法，momentum用于加速


for epoch in range(num_epochs):
    train(model,device,train_loader,optimizer,epoch)
    test(model,device,test_loader)

torch.save(model.state_dict(),"mnist_cnn.pt")
