import os

import torch
from torchvision.transforms.functional import to_pil_image

from style_transfer import StyleTransfer
from utils import load_image, load_json


def main():
    paths = {
        "setup": os.path.join("setup_files", "setup.json"),
        "content_img": os.path.join("data", "dancing.jpg"),
        "style_img": os.path.join("data", "picasso.jpg"),
    }
    setup = load_json(paths["setup"])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    content_img = load_image(paths["content_img"], DEVICE=DEVICE)
    style_img = load_image(paths["style_img"], DEVICE=DEVICE)
    style_img = style_img.to(DEVICE)
    content_img = content_img.to(DEVICE)

    input_img = torch.randn(content_img.shape)
    input_img = input_img.to(DEVICE)
    input_img.requires_grad_(True)

    style_transfer = StyleTransfer(
        content_img=content_img, style_img=style_img, setup=setup, DEVICE=DEVICE
    )

    output = style_transfer.run(input_img)

    output = to_pil_image(output.cpu().clone().squeeze(0))

    if not os.path.exists("results"):
        os.mkdir("results")

    output.save(os.path.join("results", "result.jpg"))


if __name__ == "__main__":
    main()
