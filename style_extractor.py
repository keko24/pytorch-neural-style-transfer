import torch
from torch import nn
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor

class StyleExtractor(nn.Module):
    def __init__(self, DEVICE, layers=None):
        super(StyleExtractor, self).__init__()
        if layers == None:
            layers = {
                "features.0": "conv1_1",
                "features.4": "pool1",
                "features.9": "pool2",
                "features.18": "pool3",
                "features.27": "pool4",
            }
        model = vgg19(weights='DEFAULT').to(DEVICE)
        self._layers = layers
        self._features = create_feature_extractor(model, return_nodes=layers)
        for parameter in self._features.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        return self._features(x)

    def get_layers(self):
        return self._layers
        
def compute_gram_matrix(feature_maps):
    return torch.matmul(feature_maps, torch.transpose(feature_maps, -2, -1))

def compute_loss(inputs, target, model, weights=None):
    layers = model.get_layers()
    if weights == None:
        weights = [1 / len(layers)] * len(layers)
    input_features = model(inputs)
    target_features = model(target)
    style_loss = torch.zeros(1).to(inputs.device)
    for i, layer in enumerate(layers.values()):
        input_gram = compute_gram_matrix(input_features[layer])
        target_gram = compute_gram_matrix(target_features[layer])
        layer_loss = torch.mean((input_gram - target_gram) ** 2)
        style_loss += layer_loss * weights[i]
    return style_loss
