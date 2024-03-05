import json
import os
import torch

def compute_gram_matrix(feature_maps):
    return torch.matmul(feature_maps, torch.transpose(feature_maps, -2, -1))

def compute_loss(inputs, target, model):
    input_features = model(inputs)
    target_features = model(target)
    layers = model.get_layers()
    style_loss = torch.zeros(1).to(inputs.device)
    for layer in layers.values():
        input_gram = compute_gram_matrix(input_features[layer])
        target_gram = compute_gram_matrix(target_features[layer])
        layer_loss = torch.mean((input_gram - target_gram) ** 2)
        style_loss += layer_loss * 0.2
    return style_loss

def load_json(path):
    if os.path.exists(path) and path.endswith('.json'):
        with open(path, "r") as f:
            data = json.load(f)
            return data
    else:
        print("Invalid path.")
