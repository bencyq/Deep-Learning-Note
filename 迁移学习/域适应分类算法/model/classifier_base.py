import torch.nn as nn



class Classifier(nn.Module):

    def __init__(self, backbone, num_classes, bottleneck=None, bottleneck_dim=256, head=None, finetune=True):

        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1,1)),
                nn.Flatten()
            )
            self._feature_dim = backbone.out_features

        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._feature_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._feature_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self):

        return self._feature_dim

    def forward(self, x):

        f = self.backbone(x)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self, base_lr=1.0):

        params = [
            {'params': self.backbone.parameters(), 'lr': 0.1*base_lr if self.finetune else 1.0*base_lr},
            {'params': self.bottleneck.parameters(), 'lr': 1.0*base_lr},
            {'params': self.head.parameters(), 'lr':1.0*base_lr}
        ]
        return params