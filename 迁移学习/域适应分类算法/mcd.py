### Maximum Classiﬁer Discrepancy for Unsupervised Domain Adaptation 2018
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
import torch.nn.functional as F

sys.path.append('../../..')
from utils.logger import CompleteLogger
from dataset.dataloader import get_digits_dataloader, get_dataloader
from model.feature_extractor import resnet50
from model.classifier_head import ImageClassifierHead
from utils.meter import AverageMeter, ProgressMeter
from loss.classifier_discrepancy_loss import classifier_discrepancy
from utils.accuracy import accuracy
from utils.metric import ConfusionMatrix, collect_feature, entropy
from utils import tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True

    ## 加载数据
    # train_source_loader, train_source_iter = get_digits_dataloader(args, split='train', phase='train', domain='source')
    # train_target_loader, train_target_iter = get_digits_dataloader(args, split='test', phase='train', domain='target')
    # val_loader, val_iter = get_digits_dataloader(args, split='test', phase='val', domain='target')
    # test_loader, test_iter = val_loader, val_iter
    train_source_loader, train_source_iter = get_dataloader(args, phase='train', domain='source')
    train_target_loader, train_target_iter = get_dataloader(args, phase='train', domain='target')
    val_loader, val_iter = get_dataloader(args, phase='val', domain='target')
    test_loader, test_iter = val_loader, val_iter
    ## 创建模型
    print("=> using pre-trained model '{}'".format(args.arch))
    G = resnet50(pretrained=True).to(device)
    num_classes = train_source_loader.dataset.num_classes
    #### 分类器head
    F1 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim).to(device)
    F2 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim).to(device)
    ## 定义优化算法，学习率, 损失评价
    optimizer_g = SGD(G.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = SGD([
        {'params': F1.parameters()},
        {'params': F2.parameters()},
    ], momentum=0.9, lr=args.lr, weight_decay=0.0005)

    if args.phase != 'train':
        checkpoints = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        G.load_state_dict(checkpoints['G'])
        F1.load_state_dict(checkpoints['F1'])
        F2.load_state_dict(checkpoints['F2'])

    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = G.to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, G, F1, F2, args)
        print(acc1)
        return

    ### 开始迭代训练
    best_acc1 = 0.
    best_results = None
    for epoch in range(args.epochs):
        train(train_source_iter, train_target_iter, G, F1, F2, optimizer_g, optimizer_f, epoch, args)
        results = validate(val_loader, G, F1, F2, args)
        torch.save({
            'G': G.state_dict(),
            'F1': F1.state_dict(),
            'F2': F2.state_dict(),
        }, logger.get_checkpoint_path('latest'))
        if max(results) > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = max(results)
            best_results = results
    print("best_acc1 = {:3.1f}, results = {}".format(best_acc1, best_results))
    checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    G.load_state_dict(checkpoint['G'])
    F1.load_state_dict(checkpoint['F1'])
    F2.load_state_dict(checkpoint['F2'])
    results = validate(test_loader, G, F1, F2, args)
    print("test_acc1 = {:3.1f}".format(max(results)))
    logger.close()


def train(train_source_iter, train_target_iter, G, F1, F2, optimizer_g, optimizer_f, epoch, args):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')
    progress = ProgressMeter(args.iters_per_epoch,
                             [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
                             prefix="Epoch: [{}]".format(epoch))
    G.train()
    F1.train()
    F2.train()
    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        x = torch.cat((x_s, x_t), dim=0)
        assert x.requires_grad is False
        data_time.update(time.time() - end)
        ###Step A train all networks to minimize loss on source domain,用source data训练两个分类器以及特征提取器
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        ### 关键部分
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + 0.01 * (
                    entropy(y1_t) + entropy(y2_t))
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        # Step B train classifier to maximize discrepancy最大化分类器差异
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        g = G(x)
        y_1 = F1(g)
        y_2 = F2(g)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        ### 关键部分
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + 0.01 * (
                    entropy(y1_t) + entropy(y2_t)) - \
               classifier_discrepancy(y1_t, y2_t) * args.trade_off
        loss.backward()
        optimizer_f.step()
        # Step C train genrator to minimize discrepancy 最小化分类器差异
        for k in range(args.num_k):  # 训练G时，多迭代几次
            optimizer_g.zero_grad()
            g = G(x)
            y_1 = F1(g)
            y_2 = F2(g)
            y1_s, y1_t = y_1.chunk(2, dim=0)
            y2_s, y2_t = y_2.chunk(2, dim=0)
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            mcd_loss = classifier_discrepancy(y1_t, y2_t) * args.trade_off
            mcd_loss.backward()
            optimizer_g.step()
        ### 更新各个指标
        cls_acc = accuracy(y1_s, labels_s)[0]
        tgt_acc = accuracy(y1_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(mcd_loss.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(data_loader, G, F1, F2, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1_1 = AverageMeter('Acc_1', ':6.2f')
    top1_2 = AverageMeter('Acc_2', ':6.2f')
    progress = ProgressMeter(len(data_loader),
                             [batch_time, top1_1, top1_2],
                             prefix='Test: ')
    G.eval()
    F1.eval()
    F2.eval()
    if args.per_class_eval:
        classes = data_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device)
            target = target.to(device)
            g = G(images)
            y1, y2 = F1(g), F2(g)
            acc1, = accuracy(y1, target)
            acc2, = accuracy(y2, target)
            if confmat:
                confmat.update(target, y1.argmax(1))
            top1_1.update(acc1.item(), images.size(0))
            top1_2.update(acc2.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        print(' * Acc1 {top1_1.avg:.3f} Acc2 {top1_2.avg:.3f}'.format(top1_1=top1_1, top1_2=top1_2))
        if confmat:
            print(confmat.format(classes))
        return top1_1.avg, top1_2.avg


if __name__ == '__main__':
    architecture_names = ['resnet50']
    dataset_names = ['MNIST']

    data_dir = '../examples/domain_adaptation/classification/data/office31'
    parser = argparse.ArgumentParser(description='MCD for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--root', metavar='DIR', default=data_dir,
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--center-crop', default=False, action='store_true',
                        help='whether use center crop during training')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--num-k', type=int, default=4, metavar='K',
                        help='how many steps to repeat the generator update')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mcd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    print(args)
    main(args)
