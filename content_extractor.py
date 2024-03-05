import torch
from torch import nn
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor

class ContentExtractor(nn.Module):
    def __init__(self, DEVICE, layer=None):
        super(ContentExtractor, self).__init__()
        if layer == None:
            layer = {
                "features.21": "conv4_2",
            }
        model = vgg19(weights='DEFAULT').to(DEVICE)
        self._layer = layer
        self._features = create_feature_extractor(model, return_nodes=layer)
        for parameter in self._features.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        return self._features(x)

    def get_layer(self):
        return list(self._layer.values())[0]
        
def compute_content_loss(inputs, target, model):
    input_features = model(inputs)
    target_features = model(target)
    layer = model.get_layer()
    content_loss = 0.5 * torch.sum((input_features[layer] - target_features[layer]) ** 2)
    return content_loss
