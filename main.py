import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Built_Dataset
from function import transform1, transform2, edgeV_loss
# from model import DnCNN, ADNet
from model import ADNet, PatchGAN, DnCNN
from model1 import PatchGAN, Generator, Generator1, UnetGenerator, PixelDiscriminator, Generator3, Generator2
from model1 import UnetGenerator, PixelDiscriminator
from model_cycle import Generator_cycle
from options import args
from test import imshow
from train import train_the_model, train_DnCNN

"environment setting"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 环境检测，检测cuda是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 可视化
writer = SummaryWriter("logs")
writer_1 = SummaryWriter("models")
# set the mode
if args.mode == "train":  # mode的默认值为train
    isTraining = True
else:
    isTraining = False
"prepare the dataset"
dataset = Built_Dataset(args.LR_image_dir, args.HR_image_dir, transform1, transform2)
trainData = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = Built_Dataset(args.LR_image_dir, args.HR_image_dir, transform1, transform2)
teat_trainData = DataLoader(dataset, batch_size=args.batch_size)
"built neural network"
netG = UnetGenerator(1, 1, 7).to(device)
netD = PixelDiscriminator(1).to(device)

net = DnCNN(1).to(device)

"built loss function"
cri1 = nn.MSELoss().to(device)  # MSE损失
cri2 = nn.L1Loss().to(device)  # l1损失
eloss = edgeV_loss().to(device)  # 边缘损失

"built optimizer function"
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))  # , weight_decay=1e-4)
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))  # , weight_decay=1e-4)

if isTraining:
    print("---------training----------")
    begin = time.time()
    netD, netG, loss = train_the_model(trainData, netG, netD, cri1, cri2, eloss, optimizerG, optimizerD, args.epoch,
                                       args.lr, device)
    torch.save(netG.state_dict(), 'model_injection/netG2_MDR.pth')
    # netD, loss = train_DnCNN(trainData, netG, cri1, cri2, eloss, optimizerG, args.epoch, args.lr, device)
    print(time.time() - begin)
else:
    print("---------testing----------")
    save_path = "data/test/denoise/pix2pix_injection/"
    # ours model
    net = UnetGenerator(1, 1, 5)
    # net.load_state_dict(torch.load("model/netG.pth"))
    # DnCNN model
    # net = DnCNN(1)
    net.load_state_dict(torch.load("model_injection/netG2_MDR.pth"))
    # for i in range(1, 4):
    #     # num = 7 + i
    #     num = i
    #     # imshow("./data/test/img/00" + str(num) + ".jpg", 7 + i, save_path, net)
    #     imshow("./data/test/noise/" + str(num) + ".jpg", i, save_path, net)
    for i in range(10):
        imshow("./data/noise injection/test/noise/" + str(i + 1).zfill(3) + ".jpg", i + 1, save_path, net)
