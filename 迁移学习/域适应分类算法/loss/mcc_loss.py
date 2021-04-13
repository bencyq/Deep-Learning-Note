import torch.nn as nn
import torch.nn.functional as F
import torch


def entropy(predictions, reduction='none'):
    epsilon = 1e-5
    H = -predictions*torch.log(predictions + epsilon) ## entropy(p) = - \sum_{c=1}^C p_c \log p_c
    H = H.sum(dim=1) #对各个类别
    if reduction == 'mean':
        return H.mean()
    else:
        return H


class MinimumClassConfusionLoss(nn.Module):

    def __init__(self, temperature):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits): # 输入是分类器的输出，softmax前
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits/self.temperature, dim=1) # 对dim=1任意
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size*entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1) ## B*1
        class_confusion_matrix = torch.mm(torch.transpose(predictions*entropy_weight, 1, 0), predictions) ##不同样本的权重
        class_confusion_matrix = class_confusion_matrix / class_confusion_matrix.sum(dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss