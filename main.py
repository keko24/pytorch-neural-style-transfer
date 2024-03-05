import os

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from skimage.exposure import match_histograms

from utils import load_json
from style_extractor import StyleExtractor, compute_loss


if __name__ == '__main__':
    setup = load_json(os.path.join("setup_files", "setup.json"))
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    image = Image.open(os.path.join('data', 'pebbles.jpg'))
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    img = preprocess(image)
    img = img.unsqueeze(0)
    img = img.to(DEVICE)

    white_noise = torch.randn(img.shape[1:])
    white_noise = white_noise.unsqueeze(0)
    white_noise = white_noise.to(DEVICE)
    white_noise.requires_grad_(True)

    model = StyleExtractor(DEVICE)

    for param in model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam([white_noise], lr=setup["lr"])
    for epoch in range(setup["epochs"]):
        optimizer.zero_grad()
        loss = compute_loss(img, white_noise, model)
        loss.backward()
        optimizer.step()
    
    unnormalize = transforms.Normalize(mean=(-mean / std).tolist(), std=(1.0 / std).tolist())
    white_noise = unnormalize(white_noise)

    image = transforms.Resize(256)(image)
    ref = transforms.ToTensor()(image)
    ref = ref.detach().cpu().numpy()
    white_noise = white_noise.detach().cpu().squeeze(0).numpy()
    matched = match_histograms(white_noise, ref, channel_axis=-1)
    matched = torch.from_numpy(matched) 
    if not os.path.exists('results'):
        os.mkdir('results')

    save_image(matched, os.path.join('results', 'pebbles-texture.jpg'))
