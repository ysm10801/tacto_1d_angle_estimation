import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import glob
import neptune.new as neptune

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

run = neptune.init(
    project="ysm10801/tacto",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOWUyOGQ1YS0zODhjLTRjNGItODc2OC0zNmQwZjQ1NWFhMmEifQ==",
)

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



# Dataset 상속
class TACTO_Dataset(Dataset):
    def __init__(self, datnum):
        # path_us = '/home/yang/tacto/examples/data1/'
        path_us = '/home/yang/tacto/examples/testdata/'

        self.file_list = glob.glob(path_us + '*')
        self.file_list.sort()
        
        self.x_data=[]
        self.y_data=[]

        for i in range(datnum):
            x_img = img.imread(self.file_list[i]).transpose(2,0,1)
            self.x_data.append(x_img)

            label_name = self.file_list[i].split('_')[1]
            label_name = label_name[:label_name.find('.png')]
            self.y_data.append([float(label_name)])

  # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


# datnum=30000
datnum=6000
batch_size = 64

dataset = TACTO_Dataset(datnum)

dataset_size = len(dataset)

# # Data Split (Train/Val/Test)
# train_size = int(dataset_size * 0.6)
# validation_size = int(dataset_size * 0.2)
# test_size = dataset_size - train_size - validation_size

# train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

# print(f"Training Data Size : {len(train_dataset)}")
# print(f"Validation Data Size : {len(validation_dataset)}")
print(f"Testing Data Size : {len(dataset)}")

# train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
# val_loader = DataLoader(validation_dataset, batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


# for X_train, y_train in train_loader:
#     print('X_train:', X_train.size(), 'type:', X_train.type())
#     print('y_train:', y_train.size(), 'type:', y_train.type())
#     break

# pltsize = 1
# plt.figure(figsize=(10 * pltsize, pltsize))

# for i in range(2): # Watching just 10 images
#     plt.subplot(1, 10, i + 1)
#     plt.axis('off')
#     plt.imshow(X_train[i].permute(1,2,0).cpu())
#     plt.title('angle: ' + str(y_train[i].item()))
# plt.show()


model = torchvision.models.resnet50(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1, bias=False)

model.to(device)

# print(model)

val_loss_avg = []

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                T_0 = 15,
                                                                T_mult= 2,
                                                                eta_min=1e-6)

criterion = nn.MSELoss()

nb_epochs = 100

run["algorithm"] = "ResNet_50"

PARAMS = {
    "batch_size": 64,
    "learning_rate": 0.005,
    "optimizer": "SGD",
}
run["parameters"] = PARAMS


def train(model, train_loader, optimizer):
    model.train()
    loss_train = 0
    for batch_idx, samples in enumerate(train_loader):
        # print(batch_idx)
        # print(samples)
        image, angle = samples

        image = image.to(device)
        angle = angle.to(device)

        output = model(image)
    
        # cost 계산
        loss = criterion(output, angle)
        loss_train += criterion(output, angle).item()
        # cost로 H(x) 계산
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1)%10 == 0:
            print('Train : Epoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
                            epoch, nb_epochs, batch_idx+1, len(train_loader), loss.item()))
    loss_train /= (len(train_loader.dataset) / batch_size)
    run["train/Loss"].log(loss_train)
    scheduler.step()

def evaluate(model, loader):
    model.eval()
    loss = 0
    # test_accuracy = 0
    with torch.no_grad():
        for batch_idx, samples in enumerate(loader):
            # print(batch_idx)
            # print(samples)
            image, angle = samples

            image = image.to(device)
            angle = angle.to(device)

            output = model(image)

            loss += criterion(output, angle).item()

    loss /= (len(loader.dataset) / batch_size) # 'a /= b' is equal to 'a = a/b
    run["eval/Loss"].log(loss)

    # if loader == val_loader:
    #     print("Model Output : {:.4f}, Angle : {:.4f}".format(output[0][0], angle[0][0]))
    #     print("{} :  [Epoch: {}/{}], \tValidation Loss: {:.6f}, \t \n".format(
    #         'Validation', epoch, nb_epochs, loss))
    #     val_loss_avg.append(loss)
    #     if loss <= min(val_loss_avg):
    #         torch.save(model.state_dict(), '/home/yang/tacto/examples/model_dict/model_weight.pth')
    # else:
    print("{} : \tTest Loss: {:.6f}, \t \n".format(
        'Test', loss))

    rnd_output=round(output[0, 0].tolist(), 4)
    rnd_angle=round(angle[0, 0].tolist(), 4)

    output_img = img.imread("/home/yang/tacto/examples/data1/"+str(int(rnd_output*10000))+"_"+str(rnd_output)+".png")
    angle_img = img.imread("/home/yang/tacto/examples/data1/"+str(int(rnd_angle*10000))+"_"+str(rnd_angle)+".png")
    
    plt.subplot(1,2,1)
    plt.imshow(output_img)
    plt.title("Predicted Angle Image")

    plt.subplot(1,2,2)
    plt.imshow(angle_img)
    plt.title("Correct Angle Image")

    plt.show()
        


# for epoch in range(1, nb_epochs + 1):
#     train(model, train_loader, optimizer)
#     val_loss = evaluate(model, val_loader)


model.load_state_dict(torch.load('/home/yang/tacto/examples/model_dict/model_weight.pth'))

test_loss = evaluate(model, test_loader)