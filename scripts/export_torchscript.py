# scripts/export_torchscript.py
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from mobilevos.model import MobileVOSModel

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
    
    dummy_frame = torch.randn(1, 3, 480, 854, device=device)
    dummy_prev_frame = torch.randn(1, 3, 480, 854, device=device)
    dummy_prev_mask = torch.zeros(1, 480, 854, device=device, dtype=torch.long)
    dummy_first_frame = dummy_frame.clone()
    dummy_first_mask = torch.zeros(1, 480, 854, device=device, dtype=torch.long)
    
    traced = torch.jit.trace(student, (dummy_frame, dummy_prev_frame, dummy_prev_mask, dummy_first_frame, dummy_first_mask))
    traced.save("mobilevos_student.pt")
    print("Model exported to mobilevos_student.pt")
    
if __name__ == "__main__":
    main()