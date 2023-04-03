import torch
from torchvision.models import AlexNet_Weights, ResNet50_Weights, alexnet, resnet50


class ReverseLayerF(torch.autograd.Function):
    def forward(self, x, alpha):
        self.alpha = alpha

        return x.view_as(x)

    def backward(self, grad_outputs):
        output = grad_outputs.neg() * self.alpha

        return output, None


class DANNModel(torch.nn.Module):
    def __init__(self):
        super(DANNModel, self).__init__()

        self.feature = torch.nn.Sequential(
            torch.nn.Linear(149, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(True),
        )
        
        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 5),
            torch.nn.LogSoftmax(dim=1),
        )

        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 2),
            torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, input, alpha):
        feature = self.feature(input)
        feature = feature.view(input.data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
