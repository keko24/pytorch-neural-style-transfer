import json
import os

from PIL import Image
from torchvision import transforms

preprocessor = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ToTensor(),
    ]
)


def load_image(path, DEVICE):
    content = Image.open(path)
    content = preprocessor(content)
    content = content.unsqueeze(0)
    content = content.to(DEVICE)
    return content


def load_json(path):
    if os.path.exists(path) and path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
            return data
    else:
        print("Invalid path.")
