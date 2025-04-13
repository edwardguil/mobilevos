# datasets/youtubevos.py
import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class YouTubeVOSDataset(Dataset):
    def __init__(self, root, year=2019, split="train", seq_len=5, transform=None):
        self.root = root
        self.year = year
        self.split = split
        self.seq_len = seq_len
        self.transform = transform or T.Compose([T.ToTensor()])
        images_dir = os.path.join(root, f"{year}", split, "JPEGImages")
        masks_dir = os.path.join(root, f"{year}", split, "Annotations")
        self.videos = sorted(os.listdir(images_dir))
        self.frames_dict = {}
        for vid in self.videos:
            img_paths = sorted(glob.glob(os.path.join(images_dir, vid, '*.jpg')))
            mask_paths = sorted(glob.glob(os.path.join(masks_dir, vid, '*.png')))
            if len(img_paths) == 0 or len(mask_paths) == 0:
                continue
            self.frames_dict[vid] = (img_paths, mask_paths)
    
    def __len__(self):
        return len(self.frames_dict)
    
    def __getitem__(self, idx):
        vid = list(self.frames_dict.keys())[idx]
        img_paths, mask_paths = self.frames_dict[vid]
        num_frames = len(img_paths)
        if self.seq_len is None or self.seq_len >= num_frames:
            start = 0
            seq_indices = list(range(num_frames))
        else:
            start = random.randint(0, num_frames - self.seq_len)
            seq_indices = list(range(start, start + self.seq_len))
        frames, masks = [], []
        for i in seq_indices:
            img = Image.open(img_paths[i]).convert('RGB')
            mask = Image.open(mask_paths[i])
            mask_np = np.array(mask)
            mask_np[mask_np > 0] = 1
            mask = Image.fromarray(mask_np.astype(np.uint8))
            frames.append(self.transform(img))
            mask = T.functional.pil_to_tensor(mask).squeeze(0)
            masks.append(mask.long())
        frames = torch.stack(frames, dim=0)
        masks = torch.stack(masks, dim=0)
        return frames, masks