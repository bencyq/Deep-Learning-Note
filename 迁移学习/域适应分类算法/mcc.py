import argparse
import random
import argparse
import torch
import warnings
import shutil
import time
from torch.nn import functional as F
from torch.backends import  cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import os.path as osp


from utils.logger import CompleteLogger
from dataset.dataloader import get_dataloader
from model.feature_extractor import resnet50
from model.classifier import ImageClassifier
from model.domain_classifier import DomainDiscriminator
from utils.meter import AverageMeter, ProgressMeter
from loss.adversarial_loss import DomainAdversarialLoss
from loss.mcc_loss import MinimumClassConfusionLoss
from utils.accuracy import accuracy
from utils.metric import ConfusionMatrix, collect_feature
from utils import tsne, a_distance




def main(args):

    logger = CompleteLogger(args.log, args.phase)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 加载数据
    train_source_loader, train_source_iter = get_dataloader(args, phase='train', domain='source')
    train_target_loader, train_target_iter = get_dataloader(args, phase='train', domain='target')
    val_loader, val_iter = get_dataloader(args, phase='val', domain='target')
    test_loader, test_iter = val_loader, val_iter
    ## 创建模型
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = resnet50(pretrained=True)
    num_classes = train_source_loader.dataset.num_classes
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim).to(device)
    ## 定义优化算法，学习率, 损失评价
    optimizer = torch.optim.SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    #############################################################################

    mcc = MinimumClassConfusionLoss(temperature=args.temperature) ###最关键的地方

    ##############################################################################

    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    if args.phase == 'analysis':
        pass
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, device, args)
        print('acc1: ', acc1)
        return

    #### 开始训练
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print('lr:', lr_scheduler.get_last_lr()[0])
        train(train_source_iter, train_target_iter, classifier, mcc, optimizer, lr_scheduler, epoch, device, args)
        acc1 = validate(val_loader, classifier, device, args)
        # 保存模型
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
    print("best_acc1 = {:3.1f}".format(best_acc1))

    ##在test上测试best model
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, device, args)
    print("test_acc1 = {:3.1f}".format(acc1))
    logger.close()


def train(train_source_iter, train_target_iter, model, mcc, optimizer, lr_scheduler, epoch, device, args):

    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(args.iters_per_epoch, [batch_time, data_time, losses, trans_losses, cls_accs],
                                 prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        data_time.update(time.time() - end)
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2 ,dim=0) # 预测的输出
        cls_loss = F.cross_entropy(y_s, labels_s)

        #########################################################################

        mcc_loss = mcc(y_t) ### 关键的实现

        #########################################################################
        loss = cls_loss + mcc_loss * args.trade_off
        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(mcc_loss.item(), x_s.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def validate(dataloader, model, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(dataloader),[batch_time, losses, top1, top5], prefix='Test: ')
    model.eval()
    if args.per_class_eval: ### 混淆矩阵多研究下
        classes = dataloader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(dataloader):
            images = images.to(device)
            target = target.to(device)
            #前向
            output, _ = model(images)
            loss = F.cross_entropy(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            if confmat:
                print(confmat.format(classes))
    return top1.avg


if __name__ == '__main__':
    # architecture_names = sorted(
    #     name for name in models.__dict__
    #     if name.islower() and not name.startswith("__")
    #     and callable(models.__dict__[name])
    # )
    # dataset_names = sorted(
    #     name for name in datasets.__dict__
    #     if not name.startswith("__") and callable(datasets.__dict__[name])
    # )

    data_dir = '../examples/domain_adaptation/classification/data/office31'  ## officehome
    parser = argparse.ArgumentParser(description='MCC for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--root', metavar='DIR', default=data_dir, help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', help='(default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--center-crop', default=False, action='store_true',
                        help='whether use center crop during training')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        help='backbone architecture: (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--temperature', default=2.0, type=float, help='parameter temperature scaling')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mcc',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='test', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    print(args)
    main(args)






