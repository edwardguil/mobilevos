# scripts/train_student.py
import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from mobilevos.model import MobileVOSModel
from mobilevos import losses, utils
from datasets.davis import DavisDataset
from datasets.youtubevos import YouTubeVOSDataset
from torch.utils.data import DataLoader

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    # Load student config from cfg.model (this is the default model: student)
    student_cfg = cfg.model
    # Instantiate the student network
    student = MobileVOSModel(query_encoder_name=student_cfg.query_encoder_name,
                             memory_encoder_name=student_cfg.memory_encoder_name,
                             embed_dim=student_cfg.embed_dim,
                             aspp=student_cfg.aspp,
                             memory_size=student_cfg.memory_size)
    student = student.to(device)
    
    # Load teacher for distillation using teacher configuration if provided separately.
    teacher_cfg_path = os.path.join(hydra.utils.to_absolute_path("config/model/teacher.yaml"))
    teacher_cfg = OmegaConf.load(teacher_cfg_path)
    teacher = MobileVOSModel(query_encoder_name=teacher_cfg.query_encoder_name,
                             memory_encoder_name=teacher_cfg.memory_encoder_name,
                             embed_dim=teacher_cfg.embed_dim,
                             aspp=teacher_cfg.aspp,
                             memory_size=teacher_cfg.memory_size)
    

    # Load pretrained teacher weights (teacher is fixed)
    teacher_weights = torch.load(teacher_cfg.pretrained, map_location=device)
    teacher.load_state_dict(teacher_weights)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher = teacher.to(device)
    
    # Optimizer
    optimizer = optim.Adam(student.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    
    # Create training datasets (combine DAVIS and YouTubeVOS)
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
    
    student.train()
    for epoch in range(cfg.training.epochs):
        for i, (sequences, masks) in enumerate(train_loader):
            sequences = sequences.to(device)  # [B, T, 3, H, W]
            masks     = masks.to(device)      # [B, T, H, W]
            B, T, C, H, W = sequences.shape

            # Pick a random frame t (not the first)
            t = random.randint(1, T - 1)
            cur_frames   = sequences[:,  t]
            prev_frames  = sequences[:, t-1]
            first_frames = sequences[:,  0]
            cur_gt       = masks[:,      t]
            prev_mask    = masks[:,   t-1]
            first_mask   = masks[:,      0]

            # Student forward
            rep_s, logit_s = student(
                cur_frames,
                prev_frame  = prev_frames,
                prev_mask   = prev_mask,
                first_frame = first_frames,
                first_mask  = first_mask
            )

            # Teacher forward (build memory up to t-1)
            teacher.memory_bank.reset()
            teacher.add_memory(first_frames, first_mask)
            for tm in range(1, t):
                teacher.add_memory(sequences[:, tm], masks[:, tm])
            rep_t, logit_t = teacher(cur_frames)

            # Segmentation loss
            ce_loss = losses.poly_cross_entropy_loss(logit_s, cur_gt)

            # Representation distillation
            # Downsample features by 2×
            rep_s_ds = F.avg_pool2d(rep_s, kernel_size=2, stride=2)  # [B, d, H_d, W_d]
            rep_t_ds = F.avg_pool2d(rep_t, kernel_size=2, stride=2)

            # Resize mask to match rep_s_ds spatial dims
            B, d, H_d, W_d = rep_s_ds.shape
            mask_ds = F.interpolate(cur_gt.unsqueeze(1).float(),
                                    size=(H_d, W_d),
                                    mode='nearest').squeeze(1).long()    # [B, H_d, W_d]

            # Flatten for sampling
            N = H_d * W_d
            rep_s_flat  = rep_s_ds.view(B, d, N).permute(0, 2, 1)  # [B, N, d]
            rep_t_flat  = rep_t_ds.view(B, d, N).permute(0, 2, 1)
            labels_flat = mask_ds.view(B, N)                      # [B, N]

            #   Boundary‑aware sampling
            boundary_ds = utils.get_boundary_mask(mask_ds, dilation=1).view(B, N)  # [B, N]
            M = 256
            reps_s_samples, reps_t_samples, labels_samples = [], [], []
            for b in range(B):
                b_idx  = torch.nonzero(boundary_ds[b],      as_tuple=False).view(-1)
                nb_idx = torch.nonzero(boundary_ds[b]==0,   as_tuple=False).view(-1)
                if b_idx.numel() >= M:
                    chosen = b_idx[torch.randperm(b_idx.numel(), device=b_idx.device)[:M]]
                else:
                    need = M - b_idx.numel()
                    nb_chosen = nb_idx[torch.randperm(nb_idx.numel(), device=nb_idx.device)[:need]]
                    chosen = torch.cat([b_idx, nb_chosen], dim=0)
                reps_s_samples.append(rep_s_flat[b, chosen])   # [M, d]
                reps_t_samples.append(rep_t_flat[b, chosen])
                labels_samples.append(labels_flat[b, chosen])  # [M]

            rep_s_samples = torch.cat(reps_s_samples, dim=0)   # [B*M, d]
            rep_t_samples = torch.cat(reps_t_samples, dim=0)
            labels_samples= torch.cat(labels_samples,  dim=0)  # [B*M]

            repr_loss = losses.representation_distillation_loss(
                rep_s_samples, rep_t_samples,
                labels_samples, omega=cfg.training.omega
            )

            # Logit distillation (full‑res boundary)
            boundary_full = utils.get_boundary_mask(cur_gt, dilation=1).to(device)
            logit_loss = losses.logit_distillation_loss(
                logit_s, logit_t,
                mask=boundary_full,
                temperature=cfg.training.temperature
            )

            total_loss = ce_loss + repr_loss + logit_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % cfg.training.logging_interval == 0:
                print(f"[Epoch {epoch+1}/{cfg.training.epochs}] "
                    f"[Batch {i+1}/{len(train_loader)}] "
                    f"CE: {ce_loss:.4f}  Rep: {repr_loss:.4f}  KD: {logit_loss:.4f}  Total: {total_loss:.4f}")

        # Checkpoint
        torch.save(student.state_dict(),
                f"{cfg.checkpoint_dir}/student_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()