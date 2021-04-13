from .classifier_base import Classifier as ClassifierBase
import torch.nn as nn



class ImageClassifier(ClassifierBase):

    def __init__(self, backbone, num_classes, bottleneck_dim=256, **kwargs):

        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


if __name__ == '__main__':

    import torch
    from feature_extractor import resnet50
    backbone = resnet50(pretrained=True)
    classifier = ImageClassifier(backbone, 31)
    img = torch.randn((2, 3, 224, 224))
    predictions, f = classifier(img)
    print(predictions.shape)
    print(f.shape)