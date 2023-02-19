from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import scipy.io as sio
from pytorch_wavelets import DWTForward, DWTInverse


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
            # 判断形状是否为[90, 2000],如果不是将data设置为全为1的矩阵;判断文件是否损坏
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

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
