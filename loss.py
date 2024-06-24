from torch import nn
from torch.nn.functional import mse_loss


def compute_gram_matrix(input):
    batch_size, cnn_channels, height, width = input.size()
    features = input.view(batch_size, cnn_channels, height * width)
    return features.bmm(features.transpose(1, 2)).div(cnn_channels * height * width)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = compute_gram_matrix(target).detach()

    def forward(self, x):
        gram_matrix = compute_gram_matrix(x)
        self.loss = mse_loss(gram_matrix, self.target)
        return x
