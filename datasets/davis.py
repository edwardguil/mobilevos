import os
import glob
import random
import zipfile
import requests
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

URL_480  = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip"
URL_FULL = "https://data.vision.ee.ethz.ch/cvl/DAVIS/2017/DAVIS-2017-Unsupervised-trainval-fullres.zip"

def download_and_extract(url, root):
    """Download a ZIP from `url` into `root` and extract it.
    Raises ValueError if download fails, suggesting to check the official website.
    """
    fname = url.split('/')[-1]
    fpath = os.path.join(root, fname)
    if not os.path.exists(fpath):
        print(f"Downloading {url} ...")
        try:
            with requests.get(url, stream=True) as r:
                if r.status_code == 404:
                    raise ValueError(
                        f"Download failed - file not found (404 error).\n"
                        f"The DAVIS dataset download links may have changed.\n"
                        f"Please check https://davischallenge.org/davis2017/code.html#unsupervised "
                        f"for updated download links and replace the URL in the code."
                    )
                r.raise_for_status()  # Raises HTTPError for other status codes
                with open(fpath, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Download failed with error: {str(e)}\n"
                f"The DAVIS dataset download links may have changed.\n"
                f"Please check https://davischallenge.org/davis2017/code.html#unsupervised "
                f"for updated download links and replace the URL in the code."
            ) from e
    else:
        print(f"{fname} already present, skipping download.")
    
    print(f"Extracting {fname} ...")
    with zipfile.ZipFile(fpath, 'r') as z:
        z.extractall(root)
    print("Extraction done.")

class DavisDataset(Dataset):
    def __init__(self, root, year=2017, split="train", seq_len=5,
                 transform=None, download=False, resolution="480p",
                 size=(480, 854)):
        """
        root: base path where DAVIS will be (or is) extracted.
        split: "train" or "val"
        resolution: "480p" or "fullres"
        download: if True, download & extract the official DAVIS zip.
        """
        self.root = root
        self.year = year
        self.split = split
        self.seq_len = seq_len

        self.transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])
    
        self.mask_transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])

        res_dir = "480p" if resolution.lower().startswith("480") else "Full-Resolution"

        # Official download URLs
        data_url = URL_480 if res_dir=="480p" else URL_FULL
        if download:
            download_and_extract(data_url, root)

        # After extraction, expect a top-level "DAVIS/" folder
        base = os.path.join(root, "DAVIS")
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Expected '{base}' folder after extraction, but not found.")

        # Paths inside DAVIS/
        self.img_base = os.path.join(base, "JPEGImages", res_dir)
        self.msk_base = os.path.join(base, "Annotations_unsupervised", res_dir)
        split_file   = os.path.join(base, "ImageSets", str(year), f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # Read video names
        with open(split_file, 'r') as f:
            videos = [line.strip() for line in f if line.strip()]
        self.frames_dict = {}
        for vid in videos:
            img_dir = os.path.join(self.img_base, vid)
            msk_dir = os.path.join(self.msk_base, vid)
            imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
            msks = sorted(glob.glob(os.path.join(msk_dir, "*.png")))
            if imgs and msks:
                self.frames_dict[vid] = (imgs, msks)
            else:
                print(f"Warning: missing data for video '{vid}'")

    def __len__(self):
        return len(self.frames_dict)

    def __getitem__(self, idx):
        vid = list(self.frames_dict.keys())[idx]
        img_paths, msk_paths = self.frames_dict[vid]
        N = len(img_paths)
        if self.seq_len is None or self.seq_len >= N:
            indices = list(range(N))
        else:
            start = random.randint(0, N - self.seq_len)
            indices = list(range(start, start + self.seq_len))

        frames, masks = [], []
        for i in indices:
            img = Image.open(img_paths[i]).convert("RGB")
            m = Image.open(msk_paths[i])
            m_np = np.array(m)
            m_np[m_np > 0] = 1
            m = Image.fromarray(m_np.astype(np.uint8))
            
            frames.append(self.transform(img))
            masks.append(self.mask_transform(m).squeeze(0).long())

        return torch.stack(frames), torch.stack(masks)

if __name__ == '__main__':
    dataset = DavisDataset(".\\.data", download=False, split="train")