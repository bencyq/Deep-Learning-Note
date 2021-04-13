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
from utils.accuracy import accuracy
from utils.metric import ConfusionMatrix, collect_feature
from utils import tsne, a_distance


def main(args):

    logger = CompleteLogger(args.log, args.phase)
    if args.seed is not None:
        random.seed((args.seed))
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting from checkpoints.')
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 加载数据
    source_dataloader, train_source_iter = get_dataloader(args, phase='train', domain='source')
    target_dataloader, train_target_iter = get_dataloader(args, phase='train', domain='target')
    val_dataloader, val_target_iter = get_dataloader(args, phase='val', domain='tartget')
    test_dataloader, test_target_iter = val_dataloader, val_target_iter
    ## 创建模型
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = resnet50(pretrained=True)
    classifier = ImageClassifier(backbone, source_dataloader.dataset.num_classes).to(device)
    domain_discirminator = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    ## 定义优化算法，学习率, 损失评价
    optimizer = torch.optim.SGD(classifier.get_parameters() + domain_discirminator.get_parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_schedule = LambdaLR(optimizer, lambda x: args.lr*(1. + args.lr_gamma*float(x))**(-args.lr_decay))
    domain_adv = DomainAdversarialLoss(domain_discirminator)

    ##################

    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(source_dataloader, feature_extractor, device)
        target_feature = collect_feature(target_dataloader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_dataloader, test_target_iter, classifier, device, args)
        print(acc1)
        return

    ################

    ## 开始迭代训练
    best_acc1 = 0.
    for epoch in range(args.epochs):
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer, lr_schedule, epoch, device, args) # 训练
        acc1 = validate(val_dataloader, val_target_iter, classifier, device, args) # 验证
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest')) # 保存模型
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
    print('best_acc1 = {:3.1f}'.format(best_acc1))
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_dataloader, test_target_iter, classifier, device, args)
    print('test_acc1 = {:3.1f}'.format(acc1))
    logger.close()



def train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer, lr_schedule, epoch, device, args):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(args.iter_per_epoch,  [batch_time, data_time, losses, cls_accs, domain_accs],
                             prefix='Epoch: [{}]'.format(epoch))
    classifier.train()
    end = time.time()
    for i in range(args.iter_per_epoch):
        x_s, label_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)
        x_s, x_t = x_s.to(device), x_t.to(device)
        label_s = label_s.to(device)
        data_time.update(time.time() - end)
        #前向传播,计算loss很关键
        x = torch.cat((x_s, x_t), dim=0)
        y, f = classifier(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        cls_loss = F.cross_entropy(y_s, label_s)
        ## 重点学习对抗的loss
        adv_loss = domain_adv(f_s, f_t)
        loss_total = cls_loss + args.trade_off*adv_loss
        ## 各种指标更新
        cls_acc = accuracy(y_s, label_s)[0]
        domain_acc = domain_adv.domain_discriminator_accuracy
        losses.update(loss_total.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))
        ## 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step() ## 更新
        lr_schedule.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)



def validate(dataloader, target_iter, classifier, device, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter( len(target_iter), [batch_time, losses, top1, top5], prefix='Test: ')
    classifier.eval()
    if args.per_class_eval:
        classes = dataloader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(dataloader):
            images, target = images.to(device), target.to(device)
            output, _ = classifier(images)
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

    data_dir = '../examples/domain_adaptation/classification/data/officehome'
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')

    parser.add_argument('--root', metavar='DIR', default=data_dir, help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='OfficeHome',help='dataset:(default: Office31)')
    parser.add_argument('-s', '--source', default='Ar', help='source domain(s)')
    parser.add_argument('-t', '--target', default='Cl', help='target domain(s)')
    parser.add_argument('--center_crop', default=False, action='store_true', help='whether use center crop during training')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='backbone architecture: (default: resnet50)')
    parser.add_argument('--bottleneck_dim', default=256, type=int, help='Dimension of bottleneck')
    parser.add_argument('--trade_off', default=1., type=float,help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr_decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay',default=1e-3, type=float, metavar='W', help='weight decay', dest='weight_decay')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-i', '--iter_per_epoch', default=1000, type=int,help='Number of iterations per epoch')
    parser.add_argument('-p', '--print_freq', default=100, type=int,metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=1, type=int,help='seed for initializing training. ')
    parser.add_argument('--per_class_eval', action='store_true',help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann/office31_A2W', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    from pprint import pprint
    pprint(args)
    main(args)

