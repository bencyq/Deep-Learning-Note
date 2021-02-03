# 论文：Image-to-Image Translation with Conditional Adversarial Networks


##  目标/任务： pix2pix训练和测试

 1、要求：
 
   - 论文笔记整理一份
   
   - 几种不同的归一化: BatchNorm、LayerNorm、InstanceNorm、GroupNorm
   
   - 代码3部分：数据集+模型+训练，先理解这份简单的代码，原理弄明白，自己去注释
       
       具体要求：
           
           1、模型结构自己手绘，附在笔记文档里
           
           2、训练的逻辑，自己用笔写下来，附在笔记文档里
   
   - 下节课安排：由下面三位同学来汇报你做的过程
           陈彦琦，胡俊涛，沈江
  
   - 后续我再分享一次CycleGan的做法就结束，所以下面这份官方源码一定要提前去阅读（你们还有一个月，时间足够）
  
   - 官方源码：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 整理成笔记（这部分代码量挺多的，还包括CycleGan，一周你们估计搞不定，我跟老师商量下，接下来几周，还是以这份代码为主要项目，你们先去阅读）
    
 

# 1、 数据集

- 下载数据集:
    
    - bash ./datasets/download_pix2pix_dataset.sh facades

    - bash ./datasets/download_pix2pix_dataset.sh [cityscapes, night2day, edges2handbags, edges2shoes, facades, maps]
    
    - 以facades为例：
        下载的数据集格式 {A,B}，pix2pix的训练数据要符号这样的格式
        
    - 生成这样数据集格式的脚本和要求如下：
    
    1）创建A和B 文件夹，格式/path/to/data. 
    
    2）A和B有各自的子文件夹train, val, test.比如在/path/to/data/A/train, 放置style A的图像. 在/path/to/data/B/train, 放置对应style B的图像(val, test同样).

    3）对应{A,B}图像对必须同样大小和文件名,比如 /path/to/data/A/train/1.jpg 对应 /path/to/data/B/train/1.jpg.然后用下面脚本
     
         python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data


![QQ%E6%88%AA%E5%9B%BE20201212165121.png](attachment:QQ%E6%88%AA%E5%9B%BE20201212165121.png)



```python
import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


### 1、构建数据集类 #########
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

    
dataset_name = 'facades'
img_height = 256
img_width = 256
batch_size = 4
n_cpu = 4
# 构建dataloaders
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("./datasets/%s" %dataset_name, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("./datasets/%s" %dataset_name, transforms_=transforms_, mode="val"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)
```


```python
img_dict = iter(dataloader).next()
```


```python
img_A = img_dict['A']
```


```python
img_B = img_dict['B']
```


```python
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
img = make_grid(img_A, padding=5, normalize=True)
plt.imshow(np.transpose(img, (1,2,0)))
```




    <matplotlib.image.AxesImage at 0x7fec52660410>




    
![png](output_6_1.png)
    



```python
img = make_grid(img_B, padding=5, normalize=True)
plt.imshow(np.transpose(img, (1,2,0)))
```




    <matplotlib.image.AxesImage at 0x7fec52545bd0>




    
![png](output_7_1.png)
    


# 2、模型

![QQ%E6%88%AA%E5%9B%BE20201213215318.png](attachment:QQ%E6%88%AA%E5%9B%BE20201213215318.png)




```python
import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET   生成器
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


    
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        
        super(GeneratorUNet, self).__init__()
        #### C64-C128-C256-C512-C512-C512-C512-C512
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        #### CD512-CD512-CD512-C512-C256-C128-C64
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh())

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
        
#         ds = [x, d1, d2, d3, d4, d5, d6, d7, d8]## 为了打印加上去的
#         for d in ds:
#             print(d.shape)
            
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.final(u7)
        
#         print("*"*20)
#         ups = [ u1, u2, u3, u4, u5, u6, u7,u8]## 为了打印加上去的
#         for up in ups:
#             print(up.shape) 
        return u8
        
   
```


```python
gen = GeneratorUNet()
# print(img_A.shape)
img_gen = gen(img_A)
```


```python
 ##############################
#        Discriminator
##############################

class Discriminator(nn.Module): #需要理解判别器的输入和输出
    
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
     
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters)) # 都使用InstanceNorm2d
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        #### C64-C128-C256-C512
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
        
#         print(img_input.shape)##为了看打印，加上去的
#         out = img_input.clone()
#         for i in range(len(self.model)):
#             if self.model[i].__class__.__name__.find("Conv") != -1:
#                 out = self.model[i](out)
#                 print(out.shape)   

        return self.model(img_input)

```


```python
dis = Discriminator()
dis_value = dis(img_A, img_B) 
```


```python
patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)
np.ones((4, *patch)).shape
```




    (4, 1, 16, 16)




```python
from torch.autograd import Variable
cuda =  False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
valid = Variable(Tensor(np.ones((4, *patch))), requires_grad=False)
fake =Variable(Tensor(np.zeros((4, *patch))), requires_grad=False)
```


```python
img = torch.randn(4,1,16,16)
```


```python
criterion_GAN = torch.nn.BCEWithLogitsLoss() # 我觉得用
```


```python
criterion_GAN(img, fake)
```




    tensor(0.8147)




```python
loss_fn1 = torch.nn.CrossEntropyLoss()


x_input=torch.randn(3,3)#随机生成输入 
print('x_input:\n',x_input) 
y_target=torch.tensor([1,2,0])#设置输出具体值 print('y_target\n',y_target)
loss_fn1(x_input, y_target)

```

    x_input:
     tensor([[ 1.2085, -0.7484,  1.0887],
            [ 0.0887, -0.2817, -1.5680],
            [ 0.1974,  1.0360,  0.4730]])





    tensor(2.1618)



## 3、训练模型


```python
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


epoch = 0
n_epochs = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
decay_epoch = 100
channels = 3
sample_interval = 500
checkpoint_interval = 500

os.makedirs("images/%s" %dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" %dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
# criterion_GAN = torch.nn.MSELoss()
criterion_GAN = torch.nn.BCEWithLogitsLoss() # 我觉得用
criterion_pixelwise = torch.nn.L1Loss()
# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["A"].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  训练：Training
# ----------
prev_time = time.time()

for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch["B"].type(Tensor)) #分割图
        real_B = Variable(batch["A"].type(Tensor)) # 原图

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake =Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
# 
#         print(fake.shape)
        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()
        # GAN loss
        fake_B = generator(real_A) # 真的label--->生成的分割图
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)# 真的图片label为1 
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B) # 生成的图片和real图片做L1 loss

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)#真实的
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)# 假的
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                n_epochs,
                i,
                len(dataloader),
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
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch))
```

    [Epoch 1/100] [Batch 62/127] [D loss: 0.183100] [G loss: 39.202816, pixel: 0.367359, adv: 2.466943] ETA: 0:25:08.3670544


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-17-a341a3aa0766> in <module>
        111         # Total loss
        112         loss_G = loss_GAN + lambda_pixel * loss_pixel
    --> 113         loss_G.backward()
        114         optimizer_G.step()
        115 


    ~/anaconda3/envs/pytracking/lib/python3.7/site-packages/torch/tensor.py in backward(self, gradient, retain_graph, create_graph)
        183                 products. Defaults to ``False``.
        184         """
    --> 185         torch.autograd.backward(self, gradient, retain_graph, create_graph)
        186 
        187     def register_hook(self, hook):


    ~/anaconda3/envs/pytracking/lib/python3.7/site-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables)
        125     Variable._execution_engine.run_backward(
        126         tensors, grad_tensors, retain_graph, create_graph,
    --> 127         allow_unreachable=True)  # allow_unreachable flag
        128 
        129 


    KeyboardInterrupt: 


# Visualize


```python
import matplotlib.pyplot as plt

img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_fake_B.png')
plt.imshow(img)
```


```python
img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_real_A.png')
plt.imshow(img)
```


```python
img = plt.imread('./results/facades_label2photo_pretrained/test_latest/images/100_real_B.png')
plt.imshow(img)
```


```python

```
