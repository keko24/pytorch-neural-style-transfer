import os
import time

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from utils import load_json
from style_extractor import ContentAndStyleExtractor
from loss import compute_style_and_content_loss

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()  
        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])

    def forward(self, x):
        return self.preprocess(x)

if __name__ == '__main__':
    setup = load_json(os.path.join("setup_files", "setup.json"))
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    content = Image.open(os.path.join('data', 'dancing.jpg'))
    style = Image.open(os.path.join('data', 'picasso.jpg'))
    preprocess = Preprocess()
    content = preprocess(content)
    content = content.unsqueeze(0)
    content = content.to(DEVICE, torch.float)

    style = preprocess(style)
    style = style.unsqueeze(0)
    style = style.to(DEVICE, torch.float)

    white_noise = torch.randn(content.shape) 
    white_noise = white_noise.to(DEVICE, torch.float)
    white_noise.requires_grad_(True)

    content_and_style_extractor = ContentAndStyleExtractor(DEVICE, content, style)

    content_weight = setup["alpha"]
    style_weight = setup["beta"]
    optimizer = torch.optim.LBFGS([white_noise], lr=setup["lr"])
    epoch = [0]
    while epoch[0] < setup["epochs"]:
        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                white_noise.clamp_(0, 1)
            content_loss, style_loss = compute_style_and_content_loss(white_noise, content_and_style_extractor, DEVICE)
            content_loss *= content_weight
            style_loss *= style_weight
            loss = content_loss + style_loss
            loss.backward()
            print("Epoch {}/{}....\n".format(epoch[0] + 1, setup["epochs"]),
                  "Content Loss: {:.4f}".format(content_loss.item()),
                  "Style Loss: {:.4f}\n".format(style_loss.item()),
                  "Total Loss: {:.4f}".format(loss.item()),
                  )
            epoch[0] += 1
            return content_weight * content_loss + style_weight * style_loss

        epoch_start_time = time.time()
        optimizer.step(closure)
        epoch_end_time = int(time.time() - epoch_start_time)
        print('epoch {} end time: {:02d}:{:02d}:{:02d}'.format(epoch[0] + 1, epoch_end_time // 3600,
                                                       (epoch_end_time % 3600 // 60),
                                                       epoch_end_time % 60))

    with torch.no_grad():
        white_noise.clamp_(0, 1)

    output = to_pil_image(white_noise.squeeze(0))

    if not os.path.exists('results'):
        os.mkdir('results')

    output.save(os.path.join('results', 'result.jpg'))
