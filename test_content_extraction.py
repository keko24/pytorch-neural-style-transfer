import os

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from skimage.exposure import match_histograms

from utils import load_json
from content_extractor import ContentExtractor, compute_content_loss
from style_extractor import StyleExtractor, compute_style_loss


if __name__ == '__main__':
    setup = load_json(os.path.join("setup_files", "setup.json"))
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    content = Image.open(os.path.join('data', 'original.jpg'))
    style = Image.open(os.path.join('data', 'style.jpg'))
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    content = preprocess(content)
    content = content.unsqueeze(0)
    content = content.to(DEVICE)

    style = preprocess(style)
    style = style.unsqueeze(0)
    style = style.to(DEVICE)

    print(content.shape)
    print(style.shape)

    white_noise = torch.randn(content.shape[1:])
    white_noise = white_noise.unsqueeze(0)
    white_noise = white_noise.to(DEVICE)
    white_noise.requires_grad_(True)

    content_extractor = ContentExtractor(DEVICE)
    for param in content_extractor.parameters():
        param.requires_grad_(False)

    style_extractor = StyleExtractor(DEVICE)
    for param in style_extractor.parameters():
        param.requires_grad_(False)

    alpha = setup["alpha"]
    beta = setup["beta"]
    optimizer = torch.optim.Adam([white_noise], lr=setup["lr"])
    for epoch in range(setup["epochs"]):
        optimizer.zero_grad()
        loss = alpha * compute_content_loss(content, white_noise, content_extractor) + beta * compute_style_loss(style, white_noise, style_extractor)
        loss.backward()
        optimizer.step()
    
    unnormalize = transforms.Normalize(mean=(-mean / std).tolist(), std=(1.0 / std).tolist())
    white_noise = unnormalize(white_noise)

    if not os.path.exists('results'):
        os.mkdir('results')

    save_image(white_noise, os.path.join('results', 'pebbles-texture.jpg'))
