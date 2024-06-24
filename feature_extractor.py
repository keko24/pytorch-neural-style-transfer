from torch import nn
from torchvision import transforms
from torchvision.models import vgg19

from loss import ContentLoss, StyleLoss


class ContentAndStyleExtractor(nn.Module):
    def __init__(self, content: int, style: list[int], DEVICE):
        super(ContentAndStyleExtractor, self).__init__()
        style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        content_layers = ["conv4_2"]

        style_layers.sort()

        vgg = vgg19(weights="DEFAULT").features.eval().to(DEVICE)
        vgg.requires_grad_(False)

        self.style_losses = []
        self.content_losses = []

        self.normalize = Normalize()
        self.model = nn.Sequential(self.normalize)

        pool_idx, conv_idx, relu_idx, bn_idx = (1, 1, 1, 1)
        style_loss_idx, content_loss_idx = (1, 1)

        for layer in vgg.children():
            if isinstance(layer, nn.MaxPool2d):
                name = f"pool_{pool_idx}"
                layer = nn.AvgPool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    ceil_mode=layer.ceil_mode,
                )
                pool_idx += 1
                relu_idx, conv_idx, bn_idx = (1, 1, 1)
            elif isinstance(layer, nn.Conv2d):
                name = f"conv{pool_idx}_{conv_idx}"
                conv_idx += 1
            elif isinstance(layer, nn.ReLU):
                name = f"relu{pool_idx}_{relu_idx}"
                layer = nn.ReLU(inplace=False)
                relu_idx += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{pool_idx}_{bn_idx}"
                bn_idx += 1
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            self.model.add_module(name, layer)

            if name in style_layers:
                target_style = self.model(style).detach()
                style_loss = StyleLoss(target_style)
                self.model.add_module(f"style_loss_{style_loss_idx}", style_loss)
                self.style_losses.append(style_loss)
                style_loss_idx += 1

            if name in content_layers:
                target_style = self.model(content).detach()
                content_loss = ContentLoss(target_style)
                self.model.add_module(f"content_loss_{content_loss_idx}", content_loss)
                self.content_losses.append(content_loss)
                content_loss_idx += 1

        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], StyleLoss) or isinstance(
                self.model[i], ContentLoss
            ):
                self.model = self.model[: (i + 1)]
                break

    def forward(self, x):
        return self.model(x)


class Normalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalize = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, x):
        return self.normalize(x)
