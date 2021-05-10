import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import csv
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision import transforms

IMAGE_H = 200
IMAGE_W = 200

data_transform = transforms.Compose([
    transforms.ToTensor()
])


class DogsVSCatsDataset(data.Dataset):
    def __init__(self, mode, dir):
        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = data_transform

        if self.mode == 'train':
            dir = './dog_vs_cats/train/'
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.data_size += 1
                name = file.split(sep='.')
                if name[0] == 'cat':
                    self.list_label.append(0)
                else:
                    self.list_label.append(1)
        elif self.mode == 'test1':
            dir = './dog_vs_cats/test1/'
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.data_size += 1
                self.list_label.append(2)
            print(self.list_img)
            self.list_img.sort(key=lambda x: int(x[20:-4]))
        else:
            return print('Undefined Dataset!')

    def __getitem__(self, item):
        if self.mode == 'train':
            img = Image.open(self.list_img[item])
            img = img.resize((IMAGE_H, IMAGE_W))
            img = np.array(img)[:, :, :3]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])
        elif self.mode == 'test1':
            img = Image.open(self.list_img[item])
            # 重置大小
            img = img.resize((200, 200))
            img = np.array(img)[:, :, :3]
            return self.transform(img)
        else:
            print('None')

    def __len__(self):
        return self.data_size


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)

        self.f1 = nn.Linear(50 * 50 * 16, 128)
        self.f2 = nn.Linear(128, 64)
        self.f3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size()[0], -1)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)

        return F.softmax(x, dim=1)


model = torch.load('net.pkl')

dataset_dir = './dog_vs_cats'

f = open("./dog_vs_cats/sampleSubmission.csv", "w", encoding="utf-8", newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["id", "label"])

datafile_test = DogsVSCatsDataset('test1', dataset_dir)
print('Dataset loaded! length of train set is {0}'.format(len(datafile_test)))

for index in range(len(datafile_test)):
    print(datafile_test.list_img[index])

for index in range(0, 12500):
    img = datafile_test.__getitem__(index)
    img = img.unsqueeze(0)
    out = model(img)
    if out[0, 0] > out[0, 1]:  # 猫的概率大于狗
        print('the image is a cat')
        label = 0
    else:  # 猫的概率小于狗
        print('the image is a dog')
        label = 1
    print("{0} {1}".format(index + 1, label))
    print(datafile_test.list_img[index])
    # namefile = datafile_test.list_img[index].split(".")

    csv_writer.writerow([index + 1, label])
    img = Image.open(datafile_test.list_img[index])
    plt.figure('image')
    plt.imshow(img)
    plt.show()

print("ok")
