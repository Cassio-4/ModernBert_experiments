import json
import os

for f in os.listdir("configs/"):
    if f.endswith(".json"):
        with open(os.path.join("configs/", f), 'r') as file:
            config = json.load(file)
            