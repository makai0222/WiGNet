## 代码使用简介

1. 在`train.py`脚本中将`--data_path`设置成文件夹路径
2. 在`train.py`脚本中将`--weights`参数设成预训练权重路径或着设置为空
3. 设置好数据集的路径`--data_path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
4. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
5. `my_dataset.py`脚本设置了数据读取方式，植入了2D-DWT
6. `confusion_matrix.py`画混淆矩阵，计算评估指标
7. `normalize.py`计算数据集的均值、方差，训练时使用。

