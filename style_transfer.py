import time

import torch

from feature_extractor import ContentAndStyleExtractor


class StyleTransfer:
    def __init__(self, content_img, style_img, setup, DEVICE):
        self.DEVICE = DEVICE
        self.setup = setup
        self.model = ContentAndStyleExtractor(content_img, style_img, DEVICE=DEVICE)

    def run(self, input_img):
        content_weight = self.setup["alpha"]
        style_weight = self.setup["beta"]
        optimizer = torch.optim.LBFGS([input_img])
        epoch = [0]
        while epoch[0] < self.setup["epochs"]:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                self.model(input_img)

                style_loss, content_loss = (0, 0)

                for sl in self.model.style_losses:
                    style_loss += sl.loss

                for cl in self.model.content_losses:
                    content_loss += cl.loss

                content_loss *= content_weight
                style_loss *= style_weight
                loss = content_loss + style_loss
                loss.backward()

                epoch[0] += 1
                print(
                    "Epoch {}/{}....\n".format(epoch[0], self.setup["epochs"]),
                    "Content Loss: {:.4f}".format(content_loss.item()),
                    "Style Loss: {:.4f}\n".format(style_loss.item()),
                    "Total Loss: {:.4f}".format(loss.item()),
                )
                return content_loss + style_loss

            epoch_start_time = time.time()
            optimizer.step(closure)
            epoch_end_time = int(time.time() - epoch_start_time)
            print(
                "epoch {} end time: {:02d}:{:02d}:{:02d}".format(
                    epoch[0],
                    epoch_end_time // 3600,
                    (epoch_end_time % 3600 // 60),
                    epoch_end_time % 60,
                )
            )

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img
