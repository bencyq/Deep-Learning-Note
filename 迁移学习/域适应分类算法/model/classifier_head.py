import torch.nn as nn
import torch



class ImageClassifierHead(nn.Module):

    def __init__(self, in_features, num_classes, bottleneck_dim=1024):

        super(ImageClassifierHead, self).__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x):

        return self.head(x)