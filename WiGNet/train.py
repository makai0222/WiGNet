import os
import math
import argparse
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from wignet import WiGNet_b0 as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 先在本地端口连接服务器 ssh -L local_port:127.0.0.1:tensorboard_port username@server_port
    # 连接服务器 ssh -L 6006:127.0.0.1:6006 DuanPS@192.168.167.244
    # 服务器端输入 tensorboard --logdir=runs
    tb_writer = SummaryWriter(comment=args.summary_writer)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # Normalize的值使用normalize脚本计算
    img_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     # transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.402, 0.738, 0.732], [0.333, 0.156, 0.227])
                                     # transforms.Normalize([0.401, 0.737, 0.732], [0.335, 0.158, 0.229])
                                     ]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.402, 0.738, 0.732], [0.333, 0.156, 0.227])
                                   # transforms.Normalize([0.401, 0.737, 0.732], [0.339, 0.160, 0.232])
                                   ])
    }

    wifi_transform = {
        "train": transforms.Compose(
            [
                # transforms.Resize((15, 1000)),
                transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                transforms.Normalize([14.865301, 14.513719, 16.465462], [6.3876176, 6.3046455, 6.8047338])
            ]),
        "val": transforms.Compose(
            [
                # transforms.Resize((15, 1000)),
                transforms.Normalize([14.865301, 14.513719, 16.465462], [6.3876176, 6.3046455, 6.8047338])
            ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              img_transform=img_transform["train"],
                              wifi_transform=wifi_transform["train"],
                              )

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            img_transform=img_transform["val"],
                            wifi_transform=wifi_transform["val"],
                            )

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = create_model(num_classes=args.num_classes)
    model.to(device)

    # 如果存在预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                # if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(params=pg, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(params=pg, lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_time = time.time()
    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc <= val_acc:
            torch.save(model.state_dict(), args.weights_path)
            best_acc = val_acc

        if epoch + 1 == args.epochs:
            print('max val_accurate: {:.3f}'.format(best_acc))

    print(args)
    print('Finished Training')
    print('used time : {}'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5E-4)
    parser.add_argument('--weights_path', type=str,
                        default="./weights/efficientnet-b0-wifiGesturesRoom1-am1.pth")
    parser.add_argument('--summary_writer', type=str, default="efficientnet-b0-wifiGesturesRoom1-am1")
    parser.add_argument('--data_path', type=str,
                        default="../Data/csv/gestures-room1")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
