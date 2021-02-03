from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import torchvision.transforms as transforms
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

workspace_dir = 'D:\week12'


class FaceDataset(Dataset):  # 用Dataset自定义一个数据集包装类
    """
        通过继承抽象类torch.utils.data.dataset重新定义一个自己的数据集包装类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作
        实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
        p.s.必须重写__getitem__和__len__方法
        """

    def __init__(self, fnames, transform):  # 构造方法：传入文件名和自定义的transform
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)  # 定义num_samples传入数据量（图片的张数）

    def __getitem__(self, idx):  # 通过给定的[]来索引对应的data，支持从0到len(self)的索引
        fname = self.fnames[idx]

        """ 
        cv2.imread()读取图片后已多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引，具体图像的通道数由图片的格式来决定
        imread函数有两个参数，第一个参数是图片路径，第二个参数表示读取图片的形式
        """
        img = cv2.imread(fname)  # 读取图片
        img = self.BGR2RGB(img)  # 调用自定义的函数BGR2RGB将读入的BGR图片转为RGB图片
        img = self.transform(img)  # 更改cv2.imread读取图片后，存放的多维数组的格式
        return img

    def __len__(self):
        return self.num_samples  # 返回图片数量

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 通过cv2.cvtColor将读入的图片从BGR转为RGB


def get_dataset(root):  # 定义函数用来获取数据集包装类
    fnames = glob.glob(os.path.join(root, '*'))  # 获取给定路径下的所有文件夹名称
    """
    transforms.Compose函数就是将transforms组合在一起；而每一个transforms都有自己的功能。最终只要使用定义好的train_transformer 就可以按照循序处理transforms的要求的。
    """
    transform = transforms.Compose(
        [transforms.ToPILImage(),  # 将 shape 为 (C,H,W) 的 Tensor 或 shape 为 (H,W,C) 的 numpy.ndarray 转换成 PIL.Image，值不变
         transforms.Resize((64, 64)),  # 将图片缩放成50*50
         transforms.ToTensor(),  # 转化为Tensor格式，可直接进入神经网络
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    """
    那transform.Normalize()是怎么工作的呢？以上面代码为例，ToTensor()能够把灰度范围从0-255变换到0-1之间，而后面的transform.Normalize()则把0-1变换到(-1,1).
    具体地说，对每个通道而言,transform.Normalize()执行以下操作：image=(image-mean)/std
    """
    dataset = FaceDataset(fnames, transform)
    return dataset


def same_seeds(seed):  # 设置相同的种子，保证相同输入的输出是固定的
    torch.manual_seed(seed)  # 设定生成随机数的种子，并返回一个 torch._C.Generator  对象. 参数： seed (int or long)：种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置种子用于生成随机数，以使得结果是确定的
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    np.random.seed(seed)  # 用种子生成随机数
    random.seed(seed)
    """
    大部分情况下，设置这个 torch.backends.cudnn.benchmark = True 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
    如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    """
    torch.backends.cudnn.benchmark = False
    """将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，可以保证每次运行网络的时候相同输入的输出是固定的"""
    torch.backends.cudnn.deterministic = True


def weights_init(m):  # 初始化权重和偏移
    classname = m.__class__.__name__  # 获取类的名字
    """find() 方法检测字符串中是否包含子字符串str,如果包含子字符串则返回开始的索引值，否则返回-1"""
    if classname.find('Conv') != -1:  # 确认classname里不包含Conv
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # 确认classname里不包含BathNorm
        m.weight.data.normal_(1.0, 0.02)  # 初始化w（权重）
        m.bias.data.fill_(0)  # 初始化bias（偏移）


"""
    利用nn.Module定义自己的网络
需要继承nn.Module类，并实现forward方法，在构造函数中也要调用Module的构造函数
一般把网络中具有可学习参数的层放在构造函数__init__()中。
不具有可学习参数的层（如ReLU）可放在构造函数中，也可不放在构造函数中（而在forward中使用nn.functional来代替）。
可学习参数放在构造函数中，并且通过nn.Parameter()使参数以parameters（一种tensor,默认是自动求导）的形式存在Module中，并且通过parameters()或者named_parameters()以迭代器的方式返回可学习参数。
只要在nn.Module中定义了forward函数，backward函数就会被自动实现（利用Autograd)。而且一般不是显式的调用forward(layer.forward), 而是layer(input), 会自执行forward().
"""
"""
    nn.Sequential
一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
与nn.Module的主要区别在于，nn.Sequential比较简单，直接传入各个nn模块就行，nn.Module需要重写构造方法和forward函数来更加仔细的调整模型。
总而言之，nn.Sequential更简单和方便，而nn.Module的可操作空间更多，对forward运算的调整更加仔细，也更灵活
"""
"""
    Batchnorm
Batchnorm是深度网络中经常用到的加速神经网络训练，加速收敛速度及稳定性的算法，可以说是目前深度网络必不可少的一部分。
Batchnorm是归一化的一种手段，极限来说，这种方式会减小图像之间的绝对差异，突出相对差异，加快训练速度。所以说，并不是在深度学习的所有领域都可以使用BatchNorm
优点：
没有它之前，需要小心的调整学习率和权重初始化，但是有了BN可以放心的使用大学习率，但是使用了BN，就不用小心的调参了，较大的学习率极大的提高了学习速度，
Batchnorm本身上也是一种正则的方式，可以代替其他正则方式如dropout等
另外，个人认为，batchnorm降低了数据之间的绝对差异，有一个去相关的性质，更多的考虑相对差异性，因此在分类任务上具有更好的效果。
缺点：
在image-to-image这样的任务中，尤其是超分辨率上，图像的绝对差异显得尤为重要，所以batchnorm的scale并不适合。
"""


class Generator(nn.Module):  # 定义G生成器
    def __init__(self, in_dim, dim=64):  # dim为维度
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):  # 自定义层的模型
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),  # 逆卷积
                nn.BatchNorm2d(out_dim),  # nn.BatchNorm2d()的作用是根据统计的mean和var来对数据进行标准化
                nn.ReLU())  # 对结果进行激活

        self.l1 = nn.Sequential(  # 第一层
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),  # 第二层
            dconv_bn_relu(dim * 4, dim * 2),  # 第三层
            dconv_bn_relu(dim * 2, dim),  # 第四层
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),  # 第四层
            nn.Tanh())  # 激活
        self.apply(weights_init)  # 初始化w和b

    def forward(self, x):  # 自定义前向运算
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):  # 定义D判别器
    """
    input (N, 3, 64, 64)
    output (N, )
    """

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):  # 自定义层的函数
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))  # 对函数激活

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),  # 卷积
            conv_bn_lrelu(dim, dim * 2),  # 卷积
            conv_bn_lrelu(dim * 2, dim * 4),  # 卷积
            conv_bn_lrelu(dim * 4, dim * 8),  # 卷积
            nn.Conv2d(dim * 8, 1, 4),  # 卷积
            nn.Sigmoid())  # 激活
        self.apply(weights_init)

    def forward(self, x):  # 定义前向运算
        y = self.ls(x)
        y = y.view(-1)
        return y


"""--------------------------------------超参-----------------------------------------"""
batch_size = 64  # 样本量的大小为64
z_dim = 100  # 维度100
"""通过更改学习率避免mode collapse，当使用大的Batch Size时可以适当调高学习率"""
"""注：学习率为1e-4、batch_size为64的时候，epoch跑到12轮就出现了mode__collapse"""
lr = 1e-4  # 学习率
n_epoch = 10  # 计算次数
save_dir = os.path.join(workspace_dir, 'logs')  # 保存路径
os.makedirs(save_dir, exist_ok=True)  # 创建多层目录

# model
G = Generator(in_dim=z_dim).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()

# loss criterion
criterion = nn.BCELoss()  # nn.BCELoss()用于计算二分类问题的loss

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))  # 定义一个Adam算法的优化器
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
same_seeds(999)  # 定义同一个种子

# 数据处理
dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # shuffle=True代表是先打乱数据再取样本

# 打一张图试试看
plt.imshow(dataset[10].numpy().transpose(1, 2, 0))
# plt.show()


print(torch.cuda.is_available())  # 确认cuda可以用
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设置用Gpu的cuda来运算
print(device)  # 确认运算的设备
# for logging
z_sample = Variable(torch.randn(100, z_dim)).cuda()  # 将内存中的数据复制到显存里去，可以通过GPU进行计算

if __name__ == '__main__':  # 多线程读取要在main函数里进行，否则会报错 num_workers = 4

    # 训练代码
    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()  # imgs的内容放入显存进行运算

            bs = imgs.size(0)

            """ Train D """
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)

            # label
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()

            # dis
            """当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；
            或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播"""
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            # compute loss 计算loss
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # update model 更新参数
            D.zero_grad()  # 梯度清零
            loss_D.backward()  # 反向传播
            opt_D.step()  # 更新参数

            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)

            # compute loss
            loss_G = criterion(f_logit, r_label)

            # update model
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            print(
                f'\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',
                end='')
        G.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')  # 确定保存的路径
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)  # 直接保存Tensor为图片
        print(f' | Save some samples to {filename}.')  # 保存图片
        # show generated image
        """make_grid的作用是将若干幅图像拼成一幅图像。其中padding的作用就是子图像与子图像之间的pad有多宽"""
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.savefig('D:\\week12\\Epoch{}.png'.format(epoch))  # 保存图像
        G.train()
        if (e + 1) % 5 == 0:  # 保存参数到字典
            torch.save(G.state_dict(), os.path.join(workspace_dir, f'dcgan_g.pth'))
            torch.save(D.state_dict(), os.path.join(workspace_dir, f'dcgan_d.pth'))

    # 加载已训练好的模型
    G = Generator(z_dim)
    G.load_state_dict(torch.load(os.path.join(workspace_dir, 'dcgan_g.pth')))  # 加载已经保存的参数
    G.eval()
    G.cuda()

    # 更新图像并保存结果
    n_output = 300
    z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0
    save_dir = os.path.join(workspace_dir, 'logs')
    filename = os.path.join(save_dir, f'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    # 展示图像
    grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


"""---------------心得体会--------------"""
"""
        什么是mode collapse
    Mode collapse 是指 GAN 生成的样本单一，其认为满足某一分布的结果为 true，其他为 False，导致以上结果。
    自然数据分布是非常复杂，且是多峰值的（multimodal）。也就是说数据分布有很多的峰值（peak）或众数（mode）。每个 mode 都表示相似数据样本的聚集，但与其他 mode 是不同的。
    在 mode collapse 过程中，生成网络 G 会生成属于有限集 mode 的样本。当 G 认为可以在单个 mode 上欺骗判别网络 D 时，G 就会生成该 mode 外的样本。

    Gan网络训练次数过多容易产生mode collapse，也就是模型崩塌。
    当我的学习率为1e-4、batch_size为64的时候，epoch跑到12轮就出现了mode__collapse，具体图片可见文件夹中，第12轮产生的图片的特征已经归一化了。
    所以我适当调低了学习率为1e-5，但是由于epoch调成了100，至今还没运算完。
"""
