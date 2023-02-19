import cv2
import scipy.io as sio
from PIL import Image
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_wavelets import DWTForward, DWTInverse

# # 画loss曲线
# data = pd.read_csv("wifiGesturesRoom1-am1-tag-train_loss.csv")
# # print(data)
# x = np.array(data["time"])
# x1 = np.array(data["time1"])
# y = np.array(data["loss"])
# y1 = np.array(data["loss1"])
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# plt.plot(x, y, "b", ms=10, label="不使用2D-DWT")
# plt.plot(x1, y1, "r", ms=10, label="使用2D-DDWT")
# # plt.xticks(rotation=45)
# plt.xlabel("time/s")
# plt.ylabel("loss")
# plt.legend(loc="upper right")
# plt.savefig("9.jpg")
# # plt.show()


# # 画2D-DWT的频率能量图
# image = Image.open("C:/Users/hp/Desktop/widar3-room3/slide2.png")
# image = np.asarray(image)
# image = torch.FloatTensor(image)
# image = image.permute(2, 0, 1)
# image = torch.unsqueeze(image, 0)
# xfm = DWTForward(J=1, mode='zero', wave='db2')
# yl, yh = xfm(image)
# image = torch.squeeze(yl, 0)
# image = image.permute(1, 2, 0)
# image = np.asarray(image)
# image = Image.fromarray(np.uint8(image))
# image.show()
# image.save("slide2.png")


arrays = [0.9977, 0.9999, 0.9981, 0.9988, 0.9989, 0.9982, 0.9988, 0.9993, 0.9998,
          0.9954, 0.9983, 0.9993, 0.9995, 0.9987, 0.9998, 0.9997,
          0.9999, 0.9998, 0.9986, 0.9576, 0.9973, 0.9999, 0.9950, 0.9999, 0.9951,
          0.9984, 0.9865, 0.9991, 0.3383, 0.9998, 0.7559, 0.9992,
          0.9787, 0.7780, 0.9956, 0.9997, 0.9996, 0.9914, 0.9991, 0.9063, 0.9601,
          0.9971, 0.9997, 0.6844, 0.9944, 0.9993, 0.9997, 0.9990,
          0.9994, 0.9990, 0.9995, 0.9830, 0.9949, 0.9762, 0.9986, 0.9907, 0.9943,
          0.9832, 0.9838, 0.9625, 0.8828, 0.9797, 0.9816, 0.9977,
          0.9979, 0.9996, 0.9997, 0.9931, 0.8244, 0.9986, 0.9929, 0.9906, 0.9175,
          0.9968, 0.9990, 0.9957, 0.9886, 0.9997, 0.9995, 0.9525,
          0.9213, 0.9854, 0.9986, 0.9967, 0.9989, 0.9988, 0.9993, 0.9995, 0.9734,
          0.9922, 0.9602, 0.9989, 0.9752, 0.9954, 0.9970, 0.9974,
          0.9909, 0.9737, 0.9919, 0.9977, 0.8303, 0.9962, 0.9866, 0.9995, 0.9973,
          0.9963, 0.9955, 0.9764, 0.9999, 0.8188, 0.9947, 0.9990,
          0.9782, 0.6803, 0.9993, 0.9240, 0.7377, 0.9933, 0.9968, 0.9997, 0.9999,
          0.9888, 0.9981, 0.9966, 0.9986, 0.9999, 0.9880, 0.9931,
          0.9767, 0.9591, 0.9946, 0.9999, 0.9998, 0.9958, 0.8606, 0.9675, 0.9975,
          0.9732, 0.9537, 0.9993, 0.9932, 0.9982, 0.9999, 0.9995,
          1.0000, 0.9999, 1.0000, 0.9995, 0.9999, 0.9999, 0.9999, 0.9998, 1.0000,
          0.9995, 0.9998, 0.9999, 0.9997, 0.9992, 0.9997, 1.0000,
          1.0000, 0.9974, 0.9998, 0.9986, 0.9997, 1.0000, 0.9289, 0.9975, 0.7916,
          0.9457, 0.9956, 0.9924, 0.9830, 0.9982, 0.9967, 0.9977,
          0.9928, 0.9975, 0.6817, 0.9617, 0.9888, 0.9933, 0.9977, 0.9880, 0.9987,
          0.7529, 0.9949, 0.9976, 0.9633, 0.9603, 0.9990, 0.9963,
          0.9781, 0.9884, 0.9608, 0.9827, 0.9677, 0.9986, 0.9964, 0.9934, 0.9979,
          0.9737, 0.9702, 0.9761, 0.3050, 0.9604, 0.9784, 0.9996,
          0.9063, 0.9708, 0.9970, 0.9913, 0.9974, 0.9283, 0.9970]
a = torch.tensor(arrays)
# print(a.shape, a)
print(torch.mean(a), torch.var(a))
