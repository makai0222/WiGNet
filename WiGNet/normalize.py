# Compute mean and variance for training data
import json
import os
import random
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import scipy.io as sio


def read_split_data(root: str, val_rate: float = 0.):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", ".csv", ".CSV", ".mat"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label


def channel_expansion(x):
    a = x.shape[0]
    if a <= 40:
        x0 = torch.zeros(int((40 - a) / 2), 40, 40)
        x1 = torch.zeros(int((40 - a)-int((40 - a) / 2)), 40, 40)
        x = torch.cat((x0, x, x1), dim=0)
    else:
        x = x[0: 40, :, :]
    return x


def channel_extraction(x):
    a = x.shape[0]
    if a < 10:
        x0 = torch.zeros(int((10 - a) / 2), 20, 20)
        x1 = torch.zeros(int((10 - a) - int((10 - a) / 2)), 20, 20)
        x = torch.cat((x0, x, x1), dim=0)
    elif a == 11:
        x = x[1:, :, :]
    elif a == 12:
        x = x[1: 11, :, :]
    elif a == 13:
        x = torch.cat((x[1: 6, :, :], x[7: 12, :, :]), dim=0)
    elif a == 14:
        x = torch.cat((x[0: 2, :, :], x[3: 5, :, :], x[6: 8, :, :], x[9: 11, :, :], x[12:, :, :]), dim=0)
    elif a == 15:
        x = torch.cat((x[0: 2, :, :], x[3: 5, :, :], x[6: 8, :, :], x[9: 11, :, :], x[12: 14, :, :]), dim=0)
    elif a == 16:
        x = torch.cat((x[0: 1, :, :], x[2: 4, :, :], x[5: 6, :, :], x[7: 9, :, :], x[10: 11, :, :],
                       x[12: 14, :, :], x[15: 16, :, :]), dim=0)
    elif a == 17:
        x = torch.cat((x[0: 2, :, :], x[3: 4, :, :], x[5: 6, :, :], x[7: 8, :, :], x[9: 10, :, :], x[11: 12, :, :],
                       x[13: 14, :, :], x[15: 17, :, :]), dim=0)
    elif a == 18:
        x = torch.cat((x[0: 1, :, :], x[2: 3, :, :], x[4: 5, :, :], x[6: 7, :, :], x[8: 9, :, :], x[10: 11, :, :],
                       x[12: 13, :, :], x[14: 15, :, :], x[16:, :, :]), dim=0)
    elif a == 19 or a == 20:
        x = torch.cat((x[0: 1, :, :], x[2: 3, :, :], x[4: 5, :, :], x[6: 7, :, :], x[8: 9, :, :], x[10: 11, :, :],
                       x[12: 13, :, :], x[14: 15, :, :], x[16: 17, :, :], x[18: 19, :, :]), dim=0)
    elif a == 21 or a == 22:
        x = torch.cat((x[1: 2, :, :], x[3: 4, :, :], x[5: 6, :, :], x[7: 8, :, :], x[9: 10, :, :], x[11: 12, :, :],
                       x[13: 14, :, :], x[15: 16, :, :], x[17: 18, :, :], x[19: 20, :, :]), dim=0)
    elif a > 22:
        x = torch.cat((x[2: 3, :, :], x[4: 5, :, :], x[6: 7, :, :], x[8: 9, :, :], x[10: 11, :, :], x[12: 13, :, :],
                       x[14: 15, :, :], x[16: 17, :, :], x[18: 19, :, :], x[20: 21, :, :]), dim=0)
    return x


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, img_transform=None, wifi_transform=None, bvp_transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.img_transform = img_transform
        self.wifi_transform = wifi_transform
        self.bvp_transform = bvp_transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        if self.images_path[item].split(".")[-1] == "csv":
            data = pd.read_csv(self.images_path[item], header=None)
            data = data.values
            data = torch.FloatTensor(data)
            data = data.permute(1, 0)
            # # 判断形状是否为[90, 5000],如果不是将data设置为全为1的矩阵;判断文件是否损坏
            # if data.shape[0] == 90 and data.shape[1] == 2000:
            #     data = data.view(-1, 30, 2000)
            #     # data1 = torch.stack([data[0: 30, :], data[30: 60, :], data[60: 90, :]], dim=0)  # 和上面的结果一样，测试上面的是否正确
            # else:
            #     data = torch.ones([3, 30, 2000])
            data = data.view(-1, 30, data.shape[1])

            # 2D离散小波变换
            data = torch.unsqueeze(data, 0)
            xfm = DWTForward(J=1, mode='zero', wave='db2')
            yl, yh = xfm(data)
            data = torch.squeeze(yl, 0)

            if self.wifi_transform is not None:
                data = self.wifi_transform(data)

            # # 如果使用混合数据集,打开下面的注释
            # data = data.permute(1, 2, 0)  # H, W, C
            # data = data.numpy()  # 为下一步准备
            # # 混合训练时扩充尺寸和图片保持一致
            # data = cv2.copyMakeBorder(data, 85, 85, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # 上面和下面进行像素填充
            # data = torch.FloatTensor(data)
            # data = data.permute(2, 0, 1)  # C, H, W
        elif self.images_path[item].split(".")[-1] == "mat":
            data = sio.loadmat(self.images_path[item])
            data = data['velocity_spectrum_ro']
            data = cv2.copyMakeBorder(data, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            data = torch.FloatTensor(data)
            data = data.permute(2, 0, 1)
            data = channel_expansion(data)
            if self.bvp_transform is not None:
                data = self.bvp_transform(data)
        else:
            data = Image.open(self.images_path[item])
            # RGB为彩色图片，L为灰度图片
            if data.mode != 'RGB':
                raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
            if self.img_transform is not None:
                data = self.img_transform(data)
            # data = np.array(data)
            # data = cv2.copyMakeBorder(data, 0, 0, 400, 400, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # data = torch.FloatTensor(data)
            # data = data.permute(2, 0, 1)

        label = self.images_class[item]
        return data, label


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        root="E:/pythonProject/Data/data-csv/gestures-room2")
    data_transform = transforms.Compose([
                                         transforms.Resize(384),
                                         transforms.CenterCrop(384),
                                         transforms.ToTensor()
                                         # transforms.Normalize([0.685, 0.609, 0.573], [0.189, 0.202, 0.215])
                                         ])

    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label,
                              img_transform=data_transform, wifi_transform=transforms.Resize((15, 1000)))
    print(getStat(train_dataset))
