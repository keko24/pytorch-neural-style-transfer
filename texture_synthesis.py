from PIL import Image
import torch
from torchvision import transforms
from vgg_4_layers import VGG_4_layers

model = VGG_4_layers()

img = Image.open('data/original.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = preprocess(img)
img = img.unsqueeze(0)

if torch.cuda.is_available():
    img = img.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(img)

print(output)

