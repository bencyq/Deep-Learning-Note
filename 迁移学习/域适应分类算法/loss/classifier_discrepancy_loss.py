import torch


def classifier_discrepancy(y1, y2):

    return torch.mean(torch.abs(y1 - y2))