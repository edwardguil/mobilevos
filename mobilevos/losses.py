# mobilevos/losses.py
import torch
import torch.nn.functional as F

def poly_cross_entropy_loss(logits, labels, epsilon=1.0):
    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = labels.reshape(-1)
    ce = F.cross_entropy(logits_flat, labels_flat, reduction='none')
    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=-1)
        pt = probs.gather(dim=-1, index=labels_flat.unsqueeze(-1)).squeeze(-1)
    poly_term = epsilon * (1 - pt)
    loss = ce + poly_term
    return loss.mean()

def compute_correlation_matrix(z):
    return torch.matmul(z, z.t())

def representation_distillation_loss(z_s, z_t, labels=None, omega=0.9):
    c_ss = compute_correlation_matrix(z_s)
    with torch.no_grad():
        c_tt = compute_correlation_matrix(z_t)
        if labels is not None:
            if labels.dim() == 1:
                num_classes = int(labels.max().item() + 1)
                labels_onehot = F.one_hot(labels, num_classes=num_classes).float()
            else:
                labels_onehot = labels.float()
            yy = torch.matmul(labels_onehot, labels_onehot.t())
            target_c = omega * c_tt + (1 - omega) * yy
        else:
            target_c = c_tt
    eps = 1e-8
    norm_ss = torch.sum(c_ss ** 2)
    norm_align = torch.sum((c_ss * target_c) ** 2)
    loss = torch.log2(norm_ss + eps) - torch.log2(norm_align + eps)
    loss = loss / (c_ss.shape[0] + eps)
    return loss

def logit_distillation_loss(logits_student, logits_teacher, mask=None, temperature=0.1):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    if mask is not None:
        mask = mask.unsqueeze(1).expand_as(p_s)
        p_s = p_s * mask
        p_t = p_t * mask
    p_s = torch.clamp(p_s, min=1e-5, max=1-1e-5)
    p_t = torch.clamp(p_t, min=1e-5, max=1-1e-5)
    kl_map = p_t * torch.log(p_t / p_s)
    kl_per_pixel = kl_map.sum(dim=1)
    if mask is not None:
        valid_count = mask.sum()
        loss = kl_per_pixel.sum() / (valid_count + 1e-6)
    else:
        loss = kl_per_pixel.mean()
    return loss
