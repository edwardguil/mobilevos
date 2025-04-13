# mobilevos/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilevos.modules.backbone import get_backbone
from mobilevos.modules.memory import MemoryBank
from mobilevos.modules.aspp import ASPP

class MobileVOSModel(nn.Module):
    """
    MobileVOS network with query and memory encoder branches.
    The configuration (names, embed_dim, memory_size) is passed in at initialization.
    """
    def __init__(self, query_encoder_name, memory_encoder_name, embed_dim, aspp, memory_size):
        super(MobileVOSModel, self).__init__()
        # Encoders (backbones)
        self.query_encoder = get_backbone(query_encoder_name)
        self.memory_encoder = get_backbone(memory_encoder_name)
        # Assume the encoder outputs feature channels (if missing, default to 512)
        encoder_out_ch = getattr(self.query_encoder, "out_channels", 512)
        # Projection layers for key and value features
        self.key_proj = nn.Conv2d(encoder_out_ch, 64, kernel_size=1)
        self.val_proj = nn.Conv2d(encoder_out_ch, embed_dim, kernel_size=1)
        # Optional ASPP module
        self.aspp = ASPP(in_channels=embed_dim * 2, out_channels=embed_dim) if aspp else None
        # Decoder to fuse query and memory features
        dec_in_ch = embed_dim if aspp else embed_dim * 2
        self.decoder = nn.Sequential(
            nn.Conv2d(dec_in_ch, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(embed_dim, 2, kernel_size=1)
        # Memory bank
        self.memory_bank = MemoryBank(max_length=memory_size)
    
    def encode_memory(self, frame, mask):
        feat = self.memory_encoder(frame)
        key = self.key_proj(feat)
        value = self.val_proj(feat)
        if mask is not None:
            mask = F.interpolate(mask.unsqueeze(1).float(), size=key.shape[-2:], mode='nearest')
            key = key * mask
            value = value * mask
        key = F.normalize(key, p=2, dim=1)
        return key, value
    
    def encode_query(self, frame):
        feat = self.query_encoder(frame)
        key = self.key_proj(feat)
        value = self.val_proj(feat)
        key = F.normalize(key, p=2, dim=1)
        return key, value, feat

    def add_memory(self, frame, mask):
        key, value = self.encode_memory(frame, mask)
        self.memory_bank.push(key, value)

    def forward(self, query_frame, prev_frame=None, prev_mask=None,
                first_frame=None, first_mask=None):
        # If memory frames are provided, rebuild the memory bank on the fly.
        if prev_frame is not None and prev_mask is not None:
            self.memory_bank.reset()
            if first_frame is not None and first_mask is not None:
                k1, v1 = self.encode_memory(first_frame, first_mask)
                self.memory_bank.push(k1, v1)
            k_prev, v_prev = self.encode_memory(prev_frame, prev_mask)
            self.memory_bank.push(k_prev, v_prev)
        query_key, query_val, query_feat = self.encode_query(query_frame)
        mem_keys, mem_values = self.memory_bank.get_all()
        if mem_keys:
            mem_key = torch.cat([k.view(k.size(0), k.size(1), -1) for k in mem_keys], dim=2)
            mem_val = torch.cat([v.view(v.size(0), v.size(1), -1) for v in mem_values], dim=2)
            q_key = query_key.view(query_key.size(0), query_key.size(1), -1)
            affinity = torch.einsum('bck, bcm -> bkm', q_key, mem_key)
            affinity = affinity / (query_key.size(1) ** 0.5)
            affinity = F.softmax(affinity, dim=-1)
            mem_val = mem_val.permute(0, 2, 1)
            mem_read = torch.bmm(affinity, mem_val)
            B, Qpix, vd = mem_read.shape
            h, w = query_key.shape[-2:]
            mem_read = mem_read.permute(0, 2, 1).view(B, vd, h, w)
        else:
            mem_read = torch.zeros_like(query_val)
        fused = torch.cat([query_val, mem_read], dim=1)
        if self.aspp is not None:
            fused = self.aspp(fused)
        dec_feat = self.decoder(fused)
        logits = self.classifier(dec_feat)
        _, _, H_in, W_in = query_frame.shape
        logits_fullres = F.interpolate(logits,
                                       size=(H_in, W_in),
                                       mode='bilinear',
                                       align_corners=False)
        rep_feat = F.normalize(dec_feat, p=2, dim=1)
        return rep_feat, logits_fullres