import time

import torch
from torch.autograd import Variable
from tqdm import tqdm

from function import adjust_learning_rate, lsgan


def train_DnCNN(trainData, net, eloss, cri1, cri2, optimizerG, EPOCH, LR, device):
    net.train()
    for epoch in range(EPOCH):
        lr = adjust_learning_rate(epoch)
        # updata the lr of optimizerG
        for param_group in optimizerG.param_groups:
            param_group["lr"] = lr
            print("epoch =", epoch, "lr =", optimizerG.param_groups[0]["lr"])
        processBar = tqdm(trainData)
        for sample in processBar:
            LR_images = sample[0].to(device)
            HR_images = sample[1].to(device)
            LR_images = Variable(LR_images)
            HR_images = Variable(HR_images)

            fakeImg = net(LR_images)



            dLoss = lsgan(HR_images, fakeImg , cri1) # 返回损失
            dLoss.backward(retain_graph=True)

            Identity_loss = cri2(fakeImg, HR_images)
            edgloss = 0.0005 * eloss(fakeImg, HR_images)

            gLoss = Identity_loss + edgloss

            gLoss.backward()

            optimizerG.step()

            if (epoch + 1) % 10 == 0:
                torch.save(net.state_dict(), 'model_injection/netG2_MDR_epoch_%d_%d.pth' % (4, epoch))
    return net, gLoss


def train_the_model(trainData, netG, netD, cri1, cri2, eloss, optimizerG, optimizerD, EPOCH, LR, device):
    # set the model mode
    netG.train()
    netD.train()
    for epoch in range(EPOCH):
        lr = adjust_learning_rate(epoch)
        # updata the lr of optimizerG
        for param_group in optimizerG.param_groups:
            param_group["lr"] = lr
            print("epoch =", epoch, "lr =", optimizerG.param_groups[0]["lr"])
        processBar = tqdm(trainData)
        for sample in processBar:
            LR_images = sample[0].to(device)
            HR_images = sample[1].to(device)
            LR_images = Variable(LR_images)
            HR_images = Variable(HR_images)

            fakeImg = netG(LR_images)
            netD.zero_grad()
            realOut = netD(HR_images).mean()
            fakeOut = netD(fakeImg).mean()
            real = netD(HR_images)
            fake = netD(fakeImg.detach())
            dLoss = lsgan(real, fake, cri1) # 返回损失
            dLoss.backward(retain_graph=True)
            netG.zero_grad()  # 梯度归零
            Identity_loss = cri2(fakeImg, HR_images)
            edgloss = 0.0005 * eloss(fakeImg, HR_images)
            gLossGAN = 0.001 * cri1(fake, Variable(torch.ones(fake.size()).to("cuda")))
            gLoss = Identity_loss + gLossGAN + edgloss

            gLoss.backward()
            optimizerD.step()
            optimizerG.step()

            processBar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, EPOCH, dLoss.item(), gLoss.item(), realOut.item(), fakeOut.item()))
            if (epoch + 1) % 10 == 0:
                torch.save(netG.state_dict(), 'model_pix2pix_1//netG_epoch_%d_%d.pth' % (4, epoch))
    return netD, netG, gLoss

