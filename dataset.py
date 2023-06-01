import os
from io import BytesIO
from PIL import Image
import pandas as pd
import blobfile as bf
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Code from https://github.com/openai/improved-diffusion
def _list_files_recursively(data_dir, ending=["jpg", "jpeg", "png", "gif"]):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ending:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_files_recursively(full_path))
    return results

def _merge_directories(root_dir):
    images = _list_files_recursively(root_dir)
    caption_paths = [img_path.replace('jpg', 'txt') for img_path in images]
    captions = []
    for caption_path in caption_paths:
        with open(caption_path, 'r') as f:
            captions.append(f.read())
    return pd.DataFrame({"image": images, "caption": captions})

class cc30k(Dataset):
    def __init__(self, dir, transform=None):
        self.data = _merge_directories(dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["image"]
        image = Image.open(image_path).convert("RGB")
        caption = self.data.iloc[idx]["caption"]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'caption': caption}