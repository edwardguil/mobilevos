# scripts/train_teacher.py
import torch
import torch.optim as optim
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from mobilevos.model import MobileVOSModel
from datasets.davis import DavisDataset
from datasets.youtubevos import YouTubeVOSDataset
from torch.utils.data import ConcatDataset, DataLoader

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("Teacher Training Config:\n", OmegaConf.to_yaml(cfg))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    teacher_cfg = cfg.model  # If using teacher config, override manually or load different defaults.
    teacher = MobileVOSModel(query_encoder_name=teacher_cfg.query_encoder_name,
                             memory_encoder_name=teacher_cfg.memory_encoder_name,
                             embed_dim=teacher_cfg.embed_dim,
                             aspp=teacher_cfg.aspp,
                             memory_size=teacher_cfg.memory_size)
    teacher = teacher.to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    
    davis_dataset = DavisDataset(root=cfg.data.davis.root,
                                 year=cfg.data.davis.year,
                                 split=cfg.data.davis.split,
                                 seq_len=cfg.data.davis.seq_len)
    # ytvos_dataset = YouTubeVOSDataset(root=cfg.data.youtubevos.root,
    #                                   year=cfg.data.youtubevos.year,
    #                                   split=cfg.data.youtubevos.split,
    #                                   seq_len=cfg.data.youtubevos.seq_len)
    # train_dataset = ConcatDataset([davis_dataset, ytvos_dataset])
    train_loader = DataLoader(davis_dataset, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.training.num_workers, drop_last=True)
    
    teacher.train()
    for epoch in range(cfg.training.epochs):
        for i, (sequences, masks) in enumerate(train_loader):
            sequences = sequences.to(device)
            masks = masks.to(device)
            B, T, C, H, W = sequences.shape
            t = 0  # For teacher training you might use a different protocol.
            cur_frames = sequences[:, 0]
            cur_gt = masks[:, 0]
            _, logit = teacher(cur_frames)
            loss = torch.nn.functional.cross_entropy(logit, cur_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % cfg.training.logging_interval == 0:
                print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        torch.save(teacher.state_dict(), f"{cfg.checkpoint_dir}/teacher_epoch_{epoch+1}.pth")
        
if __name__ == "__main__":
    main()