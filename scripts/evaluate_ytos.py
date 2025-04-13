# scripts/evaluate_ytvos.py
import os
import torch
from mobilevos.model import MobileVOSModel
from datasets.youtubevos import YouTubeVOSDataset
from PIL import Image
import numpy as np
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student = MobileVOSModel(query_encoder_name="resnet18",
                             memory_encoder_name="mobilenetv2",
                             embed_dim=256,
                             aspp=True,
                             memory_size=2)
    checkpoint = torch.load("checkpoints/student_epoch_latest.pth", map_location=device)
    student.load_state_dict(checkpoint)
    student.eval()
    student.to(device)
    
    ytvos_val = YouTubeVOSDataset(root=cfg.data.youtubevos.root,
                                  year=cfg.data.youtubevos.year,
                                  split="valid",
                                  seq_len=None)
    output_dir = cfg.evaluation.youtubevos.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in range(len(ytvos_val)):
            frames, gt_masks = ytvos_val[idx]
            seq_name = ytvos_val.videos[idx]
            seq_out_dir = os.path.join(output_dir, seq_name)
            os.makedirs(seq_out_dir, exist_ok=True)
            frames = frames.unsqueeze(0).to(device)
            first_mask = gt_masks[0].unsqueeze(0).to(device)
            student.memory_bank.reset()
            student.add_memory(frames[:,0], first_mask)
            Image.fromarray((gt_masks[0].numpy()*255).astype(np.uint8)).save(os.path.join(seq_out_dir, "00000.png"))
            prev_mask = first_mask
            prev_frame = frames[:,0]
            for t in range(1, frames.size(1)):
                cur_frame = frames[:,t]
                _, logit = student(cur_frame, prev_frame=prev_frame, prev_mask=prev_mask,
                                   first_frame=frames[:,0], first_mask=first_mask)
                pred = (logit.argmax(dim=1).squeeze(0).cpu().numpy()).astype(np.uint8)
                fname = f"{t:05d}.png"
                Image.fromarray(pred*255).save(os.path.join(seq_out_dir, fname))
                prev_mask = torch.tensor(pred, device=device).unsqueeze(0)
                prev_frame = cur_frame
    print("YouTube-VOS Evaluation complete. Run official evaluation script for quantitative metrics.")
    
if __name__ == "__main__":
    main()