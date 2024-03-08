import torch
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
        return mse_loss(x, self.target)
 
class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = compute_gram_matrix(target).detach()
    
    def forward(self, x):
        gram_matrix = compute_gram_matrix(x)
        return mse_loss(gram_matrix, self.target)

def compute_style_and_content_loss(input_img, model, DEVICE, style_weights=None):
    if style_weights == None:
        style_weights = torch.FloatTensor([1 / model.num_style_layers] * model.num_style_layers)
    assert isinstance(style_weights, torch.FloatTensor) and len(style_weights) == model.num_style_layers, "style_weights should be a torch.FloatTensor containing a weight for each style layer."
    input_features = model(input_img)
    content_loss = model.content_loss(input_features["content"])
    style_loss = torch.zeros(1).to(DEVICE, torch.float)
    for features, sl, weight in zip(input_features["style"], model.style_losses, style_weights):
        style_loss += sl(features) * weight
    return content_loss, style_loss
