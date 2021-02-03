from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import numpy as np
import matplotlib
import os
import pandas as pd
import math
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    """
    通过继承抽象类torch.utils.data.dataset重新定义一个自己的数据集包装类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作
    实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
    p.s.必须重写__getitem__和__len__方法
    """

    def __init__(self, filepath, transform=None):
        super(MyDataset, self).__init__()
        self.img_files, self.labels = self.load_data(filepath)
        self.transform = transform

    def __getitem__(self, index):  # 通过给定的[]来索引对应的data，支持从0到len(self)的索引
        img_file, label = self.img_files[index], self.labels[index]
        img_file = img_file.replace('\\', '/')  # 将所有的‘\\’替换为'/'
        img_file = 'D:/week10' + img_file[2:]
        img = Image.open(img_file).convert('RGB')  # 通过图片的具体路径打开图片，并且转化为RGB模式。如果不用的话读出来是RGBA四通道的，其中A通道为透明通道，暂时用不到
        if self.transform:  # 采用自定义的transform来实现图片的转换，使之可以直接进入神经网络运算
            img = self.transform(img)
        return img, label

    def __len__(self):  # 返回dataset的大小
        return len(self.img_files)  # 返回了“图片文件的路径”数组的长度，即dataset的大小

    def load_data(self, filepath):
        data = pd.read_csv(filepath, index_col=0)  # 打开文件，并且将文件的第一列作为索引Nd
        img_files = data.iloc[:, 0].values  # 切出读取出的dataframe的第一列作为图像的具体路径，并将其通过‘values’方法转化为numpy的Ndarray数组
        labels = data.iloc[:, 1].values  # 切出读取出的dataframe的第二列作为图像的标签，并将其通过‘values’方法转化为numpy的Ndarray数组
        # 测试代码 print('img_files{}, labels{}'.format(img_files[0:6], labels[0:6]))
        print('load_data成功！')
        return img_files, labels


"""transforms.Compose函数就是将transforms组合在一起；而每一个transforms都有自己的功能。最终只要使用定义好的train_transformer 就可以按照循序处理transforms的要求的。"""
train_transforms = transforms.Compose([transforms.Resize((500, 500)),  # transforms.Resize将图像缩放成500*500
                                       transforms.RandomHorizontalFlip(),  # transforms.RandomHorizontalFlip():依概率p水平翻转
                                       transforms.ToTensor(),  # 转换为Tensor格式，直接可以进入神经网络
                                       ])
val_transforms = transforms.Compose([transforms.Resize((500, 500)),  # transforms.Resize将图像缩方成500*500
                                     transforms.ToTensor()  # 转换为Tensor格式，直接可以进入神经网络
                                     ])
TRAIN_ANNO = 'Classes_train_annotation.csv'  # 此为训练集的路径
VAL_ANNO = 'Classes_val_annotation.csv'  # 此为验证集的路径
CLASSES = ['Mammals', 'Birds']  # 创建一个列表，包含'Mammals'和 'Birds'元素
train_dataset = MyDataset(TRAIN_ANNO,
                          transform=train_transforms)  # 通过自定义的dataset建立一个数据集读入的dataset，并传入图像路径和自定义的transform方法
test_dataset = MyDataset(VAL_ANNO, transform=val_transforms)

# DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作),num_workers(加载数据的时候使用几个子进程)
train_loader = DataLoader(dataset=train_dataset, batch_size=16,
                          shuffle=True)  # batch_size为16，即每次取的样本量为16，然后进行迭代运算；shuffle=True代表是先打乱数据再取样本
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
data_loaders = {'train': train_loader, 'val': test_loader}


"""
    残差网络 ResNet
残差网络是针对非常深的神经网络来设计的，有时候深度会超过100层，这时候我们需要它从某一层网络获得激活，并通过跳远链接传递到更深的网路来构建能训练深度网络的ResNets
残差学习就是对于多加一层看看它新学习到了什么，以这个作为优化项。用SGD后向传播求最优解。
在特征传输的过程中，或者说在神经网络的特征提取的过程中，会出现信息的丢失，而残差网络的跳远连接（skip connection）可以起到信息补充的作用。
在比较深的网络中，解决在训练的过程中梯度爆炸和梯度消失问题。
总而言之，残差网络使深度的神经网络的实现成为可能
"""

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


# 残差单元
class Residual(nn.Module):
    """
    channnels的含义是每个卷积层中卷积核（过滤器）的数量，比如一般的RGB图片channels为3（三个通道）
    最初输入的图片样本的channels ，取决于图片类型，比如RGB；
    卷积操作完成后输出的out_channels，取决于卷积核的数量；不同的卷积核（过滤器）能检测不同特征，并决定了输出的数据有多少个通道，即决定了out_channels
    此时的out_channels也会作为下一次卷积时的卷积核的in_channels；
    卷积核中的in_channels ，刚刚2中已经说了，就是上一次卷积的out_channels ，如果是第一次做卷积，就是1中样本图片的channels
    ？？？downsample可能是stride非1的卷积或是池化。鉴于这是残差块，要保留梯度，所以应该是池化层？？？
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(Residual, self).__init__()  # 继承Module类自己微调模型
        """
        in_channels：在Conv1d（文本应用）中，即为词向量的维度；在Conv2d中，即为图像的通道数
        out_channels：卷积产生的通道数，有多少个out_channels，就需要多少个一维卷积（也就是卷积核的数量）
        kernel_size：卷积核的尺寸；卷积核的第二个维度由in_channels决定，所以实际上卷积核的大小为kernel_size * in_channels
        stride：步长
        padding：填充。对输入的每一条边，补充0的层数，能够使卷积过的图像不会过于缩小，同时放大了角落或图像边缘的信息发挥的作用
        如果padding=(kernel_size-1)/2 那么经过卷积后，输入和输出的图像大小依旧相等，如果padding=0，那么必然会缩小（当然这里默认stride——步长为1）
        """
        # 第一层卷积神经网络
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)  # 指定stride
        # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        # nn.BatchNorm2d()的作用是根据统计的mean和var来对数据进行标准化
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 第二层卷积神经网络
        self.bn2 = nn.BatchNorm2d(out_channels)
        # ？？？下采样    1.使得图像符合显示区域的大小     2.生成对应图像的缩略图    ？？？
        self.downsamples = downsample
        if self.downsamples:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    """
    forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。
    当执行model(x)的时候，底层自动调用forward方法计算结果。
    """

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))  # 调用激活函数ReLu，来激活第一层卷积运算得出的值
        out = self.bn2(self.conv2(out))  # 将激活后的值放入第二层卷积层中运算
        if self.downsamples:  # 如果可以下采样，就将x下采样，并在最后返回时采用下采样的值
            identity = self.downsample(x)

        """残差网络的计算方法就是对（初始输入+经过第一层卷积->激活->第二层卷积）进行激活，以达到跳远链接的目的"""
        return F.relu(identity + out)  # 返回初始输入和第一层运算、激活、第二层运算的总体激活值


# 残差块
def Residual_block(in_channels, out_channels, num_Residual, first_block=False):
    if first_block:  # 如果first_block为True，且输入的通道数和输出的通道数不一致，则抛出异常
        assert in_channels == out_channels
    BasicBlock = []  # 定义一个空的列表
    for i in range(num_Residual):
        if i == 0 and not first_block:  # 当进行第一次运算(i=0)且first_block=False时，添加第一个残差块，步长为2，下采样为True
            BasicBlock.append(
                Residual(in_channels, out_channels, downsample=True, stride=2))  # 执行Residual()会自动调用forward方法运算
        else:  # 不是第一次运算时，添加步长为1，下采样为False的残差块（具体见Residual的构造函数）
            BasicBlock.append(Residual(out_channels, out_channels))
    """返回一个由nn.Sequential搭建的模型，模型主要由残差单元构成"""
    return nn.Sequential(*BasicBlock)  # *BasicBlocK的’*‘代表把BasicBlock里的所有参数传进去给nn.Sequential去搭建模型


"""
Convolution卷积层之后是无法直接连接Dense全连接层的，需要把Convolution层的数据压平（Flatten），然后就可以直接加Dense层（全连接层）了。
"""


# 压平层
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)  # 将输入的数据全部压平为一维的


resnet18 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 该卷积层的参数为3*64*7(通道数)，步长为2，填充为3，偏移为0
    nn.BatchNorm2d(64),  # 数据规范化
    nn.ReLU(inplace=True),  # 参数inplace的意思是是否直接覆盖，在这里就是对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，筛选出重要的参数并保留
)
# 添加4层残差层
resnet18.add_module('layer1', Residual_block(64, 64, 2, first_block=True))  # 2*2
resnet18.add_module('layer2', Residual_block(64, 128, 2))  #
resnet18.add_module('layer3', Residual_block(128, 256, 2))
resnet18.add_module('layer4', Residual_block(256, 512, 2))
# 加入平均池化层
resnet18.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
# 定义全连接层
fc = nn.Sequential(
    FlattenLayer(),  # 压平层
    nn.Linear(512, 2)  # 线性层
)
# 加入全连接层
resnet18.add_module('fc', fc)


# 编写训练代码————————————————
# 传入参数为模型、标准、优化器、学习率调度程序、设备信息、迭代次数
def train_model(model, criterion, optimizer, scheduler, device, num_epochs=50):
    Loss_list = {'train': [], 'val': []}  # 定义字典，包含train和val两参数
    Accuracy_list_classes = {'train': [], 'val': []}  # 同上
    """
        state_dict
    torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数，state_dict作为python的字典对象将每一层的参数映射成tensor张量，
    需要注意的是torch.nn.Module模块中的state_dict只包含卷积层和全连接层的参数，当网络中存在batchnorm时，torch.nn.Module模块中的state_dict也会存放batchnorm的running_mean
    因为state_dict本质上Python字典对象，所以可以很好地进行保存、更新、修改和恢复操作（python字典结构的特性），从而为PyTorch模型和优化器增加了大量的模块化。
    """
    """
        copy.deepcopy()
        与copy.copy()不同，deepcopy()实现了深度拷贝，会将复杂对象（比如多维数组）的每一层复制出一个单独的个体来；而copy()实现的是浅层拷贝，只会拷贝第一层
    """
    best_model_wts = copy.deepcopy(model.state_dict())  # 这里是将model里的要用到的变量全部拷贝到字典，并拷贝给best_model_wts
    best_acc = 0.0
    model = model.to(device)  # 代表将模型加载到指定设备上
    print('已加载')
    for epoch in range(num_epochs):  # 进行num_epochs次训练
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print('进行训练...')
            else:
                """
                    model.eval()
                不启用 BatchNormalization 和 Dropout。训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()；
                否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。
                """
                model.eval()  # 不启用 BatchNormalization 和 Dropout
            running_loss = 0.0
            corrects_classes = 0
            """
                enumerate()
            对于一个可迭代的（iterable）、可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
            enumerate多用于在for循环中得到计数
            """
            for idx, data in enumerate(data_loaders[phase]):  # 分阶段(phase)读取数据集
                inputs, labels_classes = Variable(
                    data[0].to(device)), Variable(data[1].to(device))  # 将读取的数据转化为Variable类型，便于反向传播
                optimizer.zero_grad()  # 梯度清零
                with torch.set_grad_enabled(phase == 'train'):  # 当阶段(phase)为train时进行计算
                    x_classes = model(inputs)  # 前向传播，model(x)返回经过卷积运算的预测值
                    _, preds_classes = torch.max(x_classes, 1)  # torch.max(input,dim)返回每行或者每列的最大值
                    loss = criterion(x_classes, labels_classes)  # 求loss
                    if phase == 'train':  # 当阶段(phase)为train时进行计算
                        loss.backward()  # 反向传播
                        optimizer.step()  # 更新权值
                running_loss += loss.item() * inputs.size(0)  # 计算loss
                corrects_classes += torch.sum(preds_classes == labels_classes)  # 标记已经正确的参数？？？
            epoch_loss = running_loss / len(data_loaders[phase].dataset)  # 求得平均loss
            Loss_list[phase].append(epoch_loss)

            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_classes

            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            print('epoch {}/{}, {} Loss: {:.4f} Acc_classes:{}'.format(epoch, num_epochs - 1, phase, epoch_loss,
                                                                       epoch_acc_classes))
            if phase == 'val' and epoch_acc > best_acc:  # 在验证集里验证，并将表现最好的参数存放起来
                best_acc = epoch_acc_classes
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val classes Acc: {:.2%}'.format(best_acc))
    model.load_state_dict(best_model_wts)  # 加载最好的参数
    torch.save(model.state_dict(), 'best_model.pt')  # 保存参数模型
    print('Best val classes Acc: {:.2%}'.format(best_acc))  # 打印最准确的结果

    return model, Loss_list, Accuracy_list_classes  # 返回最好的模型、loss集、验证正确的结果集


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 在N卡的cuda上运行，如果没有cuda就在CPU上运行
print('device确定')
network = resnet18  # 选择残差网络
optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)  # 采用随机梯度下降，学习率为0.01，动量为0.9
criterion = nn.CrossEntropyLoss()  # 选择交叉熵损失函数
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                   gamma=0.1)  # 提供了基于训练中某些测量值使学习率动态下降的方法
model, Loss_list, Accuracy_list_classes = train_model(network, criterion,
                                                      optimizer, exp_lr_scheduler, device, num_epochs=50)

"""
            ————————————训练经验————————————
        不同参数设置的结果：
    层数越多非线性拟合能力越强，说白了就是能识别的图案的复杂度越高。
    层内的卷积的神经元越多，提取目标细节越丰富
    
        batch_size的设置：
    batch的size设置的不能太大也不能太小，因此实际工程中最常用的就是mini-batch，一般size设置为几十或者几百。
    对于二阶优化算法，减小batch换来的收敛速度提升远不如引入大量噪声导致的性能下降，因此在使用二阶优化算法时，往往要采用大batch哦。此时往往batch设置成几千甚至一两万才能发挥出最佳性能。
    GPU对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128…时往往要比设置为整10、整100的倍数时表现更优
    总结：
     1）batch数太小，而类别又比较多的时候，真的可能会导致loss函数震荡而不收敛，尤其是在你的网络比较复杂的时候。

     2）随着batch_size增大，处理相同的数据量的速度越快。但是也存在缺点：内存利用率提高了，但是内存容量可能撑不住了

     3）随着batch_size增大，达到相同精度所需要的epoch数量越来越多。batch_size大到一定的程度，其确定的下降方向已经基本不再变化

     4）由于上述两种因素的矛盾， Batch_Size 增大到某个时候，达到时间上的最优。

     5）由于最终收敛精度会陷入不同的局部极值，因此 Batch_Size 增大到某些时候，达到最终收敛精度上的最优。

     6）过大的batch_size的结果是网络很容易收敛到一些不好的局部最优点。同样太小的batch也存在一些问题，比如训练速度很慢，训练不容易收敛等。

     7）具体的batch_size的选取和训练集的样本数目相关。
     
     ————当我的batch_size设置为16，num_epochs设置为100时，最后最优的模型的正确率为85%，但是其实到第30次时已经基本收敛了。
     ————当我的batch_size设置为8， num_epochs设置为50 时，最后最优的模型的正确率为83%。
     ————当我的batch_size设置为32，显卡内存不足
     
        num_epochs的设置：
    没必要过多，当你发现网络算出的准确率不再提升时，就可以停止训练了
    
        learning_rate的设置：
    刚开始设置的大一些，使得下降的速度更快，后期loss_fn接近收敛的时候设置得小一点
"""

# 结果可视化
x = range(0, 50)
y1 = Loss_list["val"]
y2 = Loss_list["train"]
plt.figure(figsize=(18, 14))  # figsize:指定figure的宽和高，单位为英寸
plt.subplot(211)  # 一个figure对象包含了多个子图，可以使用subplot（）函数来绘制子图；如果是subplot （2 ，2 ，1），那么这个figure就是个2*2的矩阵图，也就是总共有4个图，1就代表了第一幅图
plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()  # 给图加上图例
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')
plt.savefig("train and val loss vs epoches.jpg")
plt.subplot(212)
y5 = Accuracy_list_classes["train"]
y6 = Accuracy_list_classes["val"]
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.legend()
plt.title('train and val Classes_acc vs. epoches')
plt.ylabel('Classes_accuracy')


# 可视化模型
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        plt.rcParams['figure.figsize'] = (8, 6)  # 显示图像的最大范围
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        fig, ax = plt.subplots(4, 4)  # 使用fig,ax = plt.subplots()将元组分解为fig和ax两个变量。
        axes = ax.flatten()  # 把父图分成2*2个子图，ax.flatten()把子图展开赋值给axes,axes[0]便是第一个子图，axes[1]是第二个
        for i, data in enumerate(data_loaders['val']):
            inputs = data[0]
            labels_classes = Variable(data[1].to(device))
            x_classes = model(Variable(inputs.to(device)))
            x_classes = x_classes.view(-1, 2)
            _, preds_classes = torch.max(x_classes, 1)
            img_input = transforms.ToPILImage()(inputs.squeeze(0))
            axes[i].imshow(img_input)  # 显示原图
            axes[i].set_title('predicted : {}\n GT:{}'.format(CLASSES[preds_classes], CLASSES[labels_classes]))
            if i == 15:
                break
        plt.suptitle('Batch1')
        plt.tight_layout()
        plt.show()


visualize_model(model)
