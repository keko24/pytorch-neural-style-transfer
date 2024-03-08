from torch import nn
from torchvision.models import vgg19

from loss import StyleLoss, ContentLoss

class ContentAndStyleExtractor(nn.Module):
    def __init__(self, DEVICE, content, style, content_layer=None, style_layers=None):
        super(ContentAndStyleExtractor, self).__init__()
        if style_layers == None:
            style_layers = [0, 5, 10, 19, 28]
        if content_layer == None:
            content_layer = 28
        
        assert isinstance(style_layers, list) and all(isinstance(layer, int) for layer in style_layers) and max(style_layers) <= 36 and min(style_layers) >= 0, "style_layes should be a list containing integer values between 0 and 36 for the indices of the layers in vgg19 that you want to use in the model."
        assert isinstance(content_layer, int) and content_layer >= 0 and content_layer <= 36, "content_layer should an integer between 0 and 36 for the layer you want to use in the model."
        style_layers.sort()

        vgg = vgg19(weights='DEFAULT').features.eval().to(DEVICE)
        for parameter in vgg.parameters():
            parameter.requires_grad_(False)

        self.content_features = nn.Sequential()

        for i in range(content_layer + 1):
            if isinstance(vgg[i], nn.ReLU):
                self.content_features.add_module(str(i), nn.ReLU(inplace=False))
            elif isinstance(vgg[i], nn.MaxPool2d):
                self.content_features.add_module(str(i), nn.AvgPool2d(kernel_size=vgg[i].kernel_size, stride=vgg[i].stride, padding=vgg[i].padding, ceil_mode=vgg[i].ceil_mode))
            else:
                self.content_features.add_module(str(i), vgg[i])

        content = self.content_features(content).detach()
        self.content_loss = ContentLoss(content)

        self.style_features = nn.ModuleList()
        self.style_losses = nn.ModuleList()
        for i, layer in enumerate(style_layers):
            layer_subset = nn.Sequential() 
            start_layer = 0 if i == 0 else style_layers[i - 1] + 1

            for j in range(start_layer, layer + 1):
                if isinstance(vgg[j], nn.ReLU):
                    layer_subset.add_module(str(j), nn.ReLU(inplace=False))
                elif isinstance(vgg[i], nn.MaxPool2d):
                    layer_subset.add_module(str(i), nn.AvgPool2d(kernel_size=vgg[i].kernel_size, stride=vgg[i].stride, padding=vgg[i].padding, ceil_mode=vgg[i].ceil_mode))
                else:
                    layer_subset.add_module(str(j), vgg[j])

            self.style_features.append(layer_subset)
            style = layer_subset(style).detach()
            style_loss = StyleLoss(style)
            self.style_losses.append(style_loss)

        self.num_style_layers = len(style_layers)

    def forward(self, x):
        content_feature_maps = self.content_features(x)
        style_feature_maps = [] 
        for layer in self.style_features:
            x = layer(x)
            style_feature_maps.append(x)
        return {"content": content_feature_maps, "style": style_feature_maps} 
