import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


# 1、构建数据集类
class ImageDataset(Dataset):
    """
    通过继承抽象类torch.utils.data.dataset重新定义一个自己的数据集包装类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作
    实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
    p.s.必须重写__getitem__和__len__方法
    """

    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))  # 查找数据集下的所有图片路径，并按照升序排序
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))  # 将test数据集下的所欲文件路径也加入到train集中去

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))  # 将读入的图片切割为左右两部分
        img_B = img.crop((w / 2, 0, w, h))
        if np.random.random() < 0.5:  # 当随机数小于0.5时，将图片反转——————————————————————————————？？？有什么用吗
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")  # [:, ::-1, :]表示将第二个通道的数据逆序反转
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)  # 转化图片的格式
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}  # 返回字典格式的数据

    def __len__(self):
        return len(self.files)  # 求得文件路径的长度


"""dataloader参数设置"""
dataset_name = 'facades'
img_height = 256
img_width = 256
batch_size = 4
n_cpu = 4  # cpu读取线程为4


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
# U-NET 生成器
##############################
"""
U-Net是德国Freiburg大学模式识别和图像处理组提出的一种全卷积结构.
和常见的先降采样到低维度，再升采样到原始分辨率的编解码(Encoder-Decoder)结构的网络相比，U-Net的区别是加入skip-connection(跳远连接，残差网络的思想)，
对应的feature maps和decode之后的同样大小的feature maps按通道拼(concatenate)一起，用来保留不同分辨率下像素级的细节信息。
U-Net相当于先降采样，再上采样，优点是对提升细节的效果非常明显
"""


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]  # 一层卷积

        """
           nn.InstanceNorm2d
        一个channel内做归一化，算H*W的均值，用在风格化迁移；
        因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。
        可以加速模型收敛，并且保持每个图像实例之间的独立。
        """
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))  # 如果normalize（标准化）参数为True，则添加归一化层
        layers.append(nn.LeakyReLU(0.2))
        """
            Dropout
        Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
        1.取平均的作用.整个dropout的过程就相当于对很多个不同的神经网络取平均,而不同的网络产生不同的过拟合,一些互为"反向"的拟合相互抵消就可以达到整体上减少过拟合.
        2.减少神经元之间复杂的共适应关系.因为dropout程序导致两个神经元不一定每次都在一个dropout中出现.这样权值的更新不再依赖于有固定关系的隐含节点的共同作用,
          阻止了某些特征仅仅在其他特征下才有效果的情况,迫使网络去学习更加鲁棒的特征.
        3.大量用于全连接层,一般设置为0.5,而在卷积网络隐藏层中由于卷积自身的稀疏化以及稀疏化的ReLu函数的大量使用等原因,Dropout策略在卷积网络隐藏层中使用较少.
        用处：明显的减少过拟合现象
        """
        if dropout:
            layers.append(nn.Dropout(dropout))  # 如果有传入dropout的值，则执行
        self.model = nn.Sequential(*layers)  # 将layers里的参数和层的信息通过nn.Sequential传入model中（自动加入了激活函数）

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),  # 进行反卷积操作
            nn.InstanceNorm2d(out_size),  # 归一化
            nn.ReLU(inplace=True),  # 激活
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)  # 建立模型

    """在深度学习处理图像时，常用的有3通道的RGB彩色图像及单通道的灰度图。张量size为cxhxw,即通道数x图像高度x图像宽度。
    在用torch.cat拼接两张图像时一般要求图像大小一致而通道数可不一致，即h和w同，c可不同。当然实际有3种拼接方式，另两种好像不常见。
    比如经典网络结构：U-Net里面用到4次torch.cat,其中copy and crop操作就是通过torch.cat来实现的。
    可以看到通过上采样（up-conv 2x2）将原始图像h和w变为原来2倍，再和左边直接copy过来的同样h,w的图像拼接。
    这样做，可以有效利用原始结构信息。"""

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)  # 将两个Tensor连接在一起
        return x


##############################
# Generator
##############################
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):  # 用U-net 生成器来进行全卷积
        super(GeneratorUNet, self).__init__()
        # C64-C128-C256-C512-C512-C512-C512-C512
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        # CD512-CD512-CD512-C512-C256-C128-C64
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 上采样
            nn.ZeroPad2d((1, 0, 1, 0)),  # 对Tensor使用0进行边界填充
            nn.Conv2d(128, out_channels, 4, padding=1),  # 卷积
            nn.Tanh())  # 激活

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        # ds = [x, d1, d2, d3, d4, d5, d6, d7, d8]## 为了打印加上去的
        # for d in ds:
        # print(d.shape)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.final(u7)
        # print("*"*20)
        # ups = [ u1, u2, u3, u4, u5, u6, u7,u8]## 为了打印加上去的
        # for up in ups:
        # print(up.shape)

        return u8


##############################
# Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2,
                                padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))  # 都使用InstanceNorm2d
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率
            return layers

        # C64-C128-C256-C512
        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        # print(img_input.shape)##为了看打印，加上去的
        # out = img_input.clone()
        # for i in range(len(self.model)):
        # if self.model[i].__class__.__name__.find("Conv") != -1:
        # out = self.model[i](out)
        # print(out.shape)
        return self.model(img_input)


"""————————————超参设置————————————"""
epoch = 0
n_epochs = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
decay_epoch = 100
channels = 3
sample_interval = 500
checkpoint_interval = 500

if __name__ == '__main__':
    # 构建dataloaders
    """ToTensor()能够把灰度范围从0-255变换到0-1之间，而后面的transform.Normalize()则把0-1变换到(-1,1)"""
    transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC), transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5)), ]  # 将图像放大尺寸，并进行BICUBIC插值，再转化为Tensor格式，然后将最大最小值缩放到（-1，1）
    dataloader = DataLoader(ImageDataset("./datasets/%s" % dataset_name, transforms_=transforms_),
                            batch_size=batch_size,
                            shuffle=True, num_workers=n_cpu, )  # 4线程读取
    val_dataloader = DataLoader(
        ImageDataset("./datasets/%s" % dataset_name, transforms_=transforms_, mode="val"), batch_size=batch_size,
        shuffle=True, num_workers=n_cpu, )

    img_dict = iter(dataloader).next()  # iter()调用可迭代对象dataloader，.next()返回下一项目值
    img_A = img_dict['A']  # 放入字典
    img_B = img_dict['B']

    img = make_grid(img_A, padding=5, normalize=True)  # 将图像拼接在一起，方便展示。是torchvision.utils里的函数
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

    """模型"""
    gen = GeneratorUNet()
    # print(img_A.shape)
    img_gen = gen(img_A)

    dis = Discriminator()
    dis_value = dis(img_A, img_B)

    patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)
    np.ones((4, *patch)).shape

    cuda = False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    valid = Variable(Tensor(np.ones((4, *patch))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((4, *patch))), requires_grad=False)

    img = torch.randn(4, 1, 16, 16)
    criterion_GAN = torch.nn.BCEWithLogitsLoss()  # 该loss 层包括了 Sigmoid 层和BCELoss 层
    criterion_GAN(img, fake)

    loss_fn1 = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数。该损失函数结合了nn.LogSoftmax()和nn.NLLLoss()两个函数
    x_input = torch.randn(3, 3)  # 随机生成输入
    print('x_input:\n', x_input)
    y_target = torch.tensor([1, 2, 0])  # 设置输出具体值 print('y_target\n',y_target)
    loss_fn1(x_input, y_target)

    """训练模型"""
    os.makedirs("images/%s" % dataset_name, exist_ok=True)  # 创建目录
    os.makedirs("saved_models/%s" % dataset_name, exist_ok=True)
    cuda = True if torch.cuda.is_available() else False
    # Loss functions
    # criterion_GAN = torch.nn.MSELoss()
    criterion_GAN = torch.nn.BCEWithLogitsLoss()  # 该loss 层包括了 Sigmoid 层和BCELoss 层
    criterion_pixelwise = torch.nn.L1Loss()  # 取预测值和真实值的绝对误差的平均数
    lambda_pixel = 100
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)  # 初始化 generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()
    if cuda:  # cuda能用的话将模型全搭载到cuda上
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
    if epoch != 0:  # 如果epoch不为0，则读取之前运算好了的模型数据
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" %
                                             (dataset_name, epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth"
                                                 % (dataset_name, epoch)))
    else:  # epoch为0，初始化权重
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers优化器
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))  # 调用Adam算法更新权重
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))  # 调用Adam算法更新权重

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def sample_images(batches_done):  # 将val数据集和G生成器生成的图片拼接在一起，并保存
        imgs = next(iter(val_dataloader))  # next()获取下一条数据，iter()调用可迭代对象的__iter__方法
        real_A = Variable(imgs["B"].type(Tensor))
        real_B = Variable(imgs["A"].type(Tensor))
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)


    # ----------
    # 训练：Training
    # ----------
    prev_time = time.time()
    for epoch in range(epoch, n_epochs):
        for i, batch in enumerate(dataloader):
            # 模型输入
            real_A = Variable(batch["B"].type(Tensor))  # 分割图
            real_B = Variable(batch["A"].type(Tensor))  # 原图
            # 正确的数据 Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
            # print(fake.shape)

            # ------------------
            # Train Generators
            # ------------------
            optimizer_G.zero_grad()  # 权重重置为零
            # GAN loss
            fake_B = generator(real_A)  # G生成器生成图片
            pred_fake = discriminator(fake_B, real_A)  # D判别器判断生成的图片和原始图片的差距
            loss_GAN = criterion_GAN(pred_fake, valid)  # 计算Gan的loss
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)  # 生成的图片和real图片做L1loss

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel  # 总的loss等于Gan网络的loss加上乘上系数的L1Loss
            loss_G.backward()  # 反向传播
            optimizer_G.step()  # 优化器更新

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()  # 优化器权重置零
            # Real loss
            pred_real = discriminator(real_B, real_A)  # 计算val数据集和train数据集的图像之间的loss(真实的loss)
            loss_real = criterion_GAN(pred_real, valid)
            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)  # 计算G生成的图片和val数据集之间的loss(假的loss)
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            # --------------
            # Log Progress
            # --------------
            # Determine approximate time left  # 计算大概的剩余时间
            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f,adv: % f] ETA: % s" % (
                    epoch,
                    n_epochs,
                    i, len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )
            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # 保存模型
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" %
                       (dataset_name, epoch))
            torch.save(discriminator.state_dict(),
                       "saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch))
