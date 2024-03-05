import json
import os

def load_json(path):
    if os.path.exists(path) and path.endswith('.json'):
        with open(path, "r") as f:
            data = json.load(f)
            return data
    else:
        print("Invalid path.")
