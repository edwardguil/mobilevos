# MobileVOS: Real-Time Video Object Segmentation via Contrastive Distillation

## Overview
This is an unofficial implementation of the MobileVOS paper. The code is provided as a research reproduction effort and may have some differences from the original implementation. While I've strived to match the paper's methodology as closely as possible, results may vary from those reported in the original work.

**MobileVOS** is a state-of-the-art, mobile/edge-friendly video object segmentation model that combines knowledge distillation and supervised contrastive learning to train a lightweight space–time–memory network. This repository re-implements the approach described in:

> **MobileVOS: Real-Time Video Object Segmentation – Contrastive Learning Meets Knowledge Distillation**  
> Roy Miles, Mehmet Kerim Yucel, Bruno Manganelli, and Albert Saá-Garriga, CVPR 2023

MobileVOS achieves competitive segmentation accuracy with dramatically fewer parameters and significantly faster runtime compared to heavyweight models.

## Implementation Features

- **End-to-End Implementation:** Contains the model architecture, full training pipeline, and evaluation on DAVIS and YouTube-VOS.
- **Distillation Framework:** Student–teacher training with both contrastive (representation) and logit distillation losses.
- **Mobile-Optimized Model:** Lightweight student network (e.g. using ResNet18/MobileNetV2 backbones) designed for resource-constrained devices.
- **Hydra-based Configuration:** All hyperparameters, paths, and model options are managed via YAML configuration files in the `config/` directory.
- **Reproducibility:** Tries to have a reproduction ready pipeline, and includes utilities for data loading, evaluation, and mobile deployment.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/edwardguil/MobileVOS.git
   cd MobileVOS
   ```

2. **Install Dependencies:**

   It is recommended to use a conda environment for containing code:
   ```bash
   conda create --name mobilevos python
   ```
   Activate the conda env
   ```
   conda activate mobilevos
   ```
   Then install dependencies with:
   ```bash
   pip install -r requirements.txt
   ```

3. **Update Configurations:**
   - In `config/model/teacher.yaml`, provide the path to pretrained teacher weights if available.

---

## Training

### Student Model Training

Run the training script using Hydra:

```bash
python scripts/train_student.py
```

Hydra reads configuration files from `config/`. You can override any parameter via command-line, for example:

```bash
python scripts/train_student.py training.batch_size=8
```

Student training combines DAVIS ~~and YouTube-VOS datasets~~ (not implemented yet). Checkpoints are saved in the directory specified by `checkpoint_dir` (default: `checkpoints/`).

### Teacher Model Training (Optional)

To train your own teacher model (if not using pretrained weights):

```bash
python scripts/train_teacher.py
```

> **Note:** The teacher model is typically a heavier model and may require longer training schedules. Use a pretrained teacher if available.

---

## Evaluation

### DAVIS Evaluation

To evaluate the trained student on DAVIS:

```bash
python scripts/evaluate_davis.py
```

This script saves segmentation masks to `results/DAVIS/`. Then, run the official DAVIS evaluation toolkit to compute J&F scores.

### YouTube-VOS Evaluation

To evaluate on YouTube-VOS:

```bash
python scripts/evaluate_ytvos.py
```

Predicted masks are saved to `results/YTVOS/`. Use the official evaluation tools to measure performance.

---

## Model Export for Mobile

Export the trained student model to TorchScript for mobile deployment:

```bash
python scripts/export_torchscript.py
```

This generates `mobilevos_student.pt`, which can be deployed using PyTorch Mobile.

---

## Configuration & Hydra

All settings—including model parameters, training hyperparameters, and data paths—are defined in YAML files in the `config/` directory:
- `config/config.yaml` is the main configuration.
- `config/data.yaml` holds dataset parameters.
- `config/training.yaml` holds training hyperparameters.
- `config/model/student.yaml` and `config/model/teacher.yaml` configure model details.

Override any parameters from the command line:

```bash
python scripts/train_student.py training.learning_rate=5e-6 data.davis.root="/new/path/to/DAVIS"
```

For more on Hydra, refer to the [Hydra Documentation](https://hydra.cc/docs/intro/).

---

## Model Architecture & Losses

**MobileVOS Model Architecture:**
- **Encoders:** Separate query and memory encoders (e.g., ResNet18 for student; ResNet50/MobileNet for teacher).
- **ASPP Module:** Multi-scale feature extractor.
- **Memory Bank:** Stores key/value representations from past frames.
- **Decoder:** Fuses query and memory features to produce segmentation.
  
**Loss Functions:**
- **Poly Cross-Entropy Loss:** Standard cross-entropy augmented with a term penalizing high-confidence mistakes.
- **Representation Distillation Loss:** A contrastive loss aligning self-similarity (correlation) matrices between student and teacher representations. Blends teacher’s correlations and ground truth information.
- **Logit Distillation Loss:** KL divergence computed on boundary pixels (identified via a Sobel-based boundary mask) to encourage the student to mimic the teacher's output.

These components follow the MobileVOS paper exactly.

---

## Citation

If you use this code in your research, please cite the orignal authors:

```bibtex
@InProceedings{Miles2023MobileVOS,
  title={MobileVOS: Real-Time Video Object Segmentation -- Contrastive Learning Meets Knowledge Distillation},
  author={Miles, Roy and Yucel, Mehmet Kerim and Manganelli, Bruno and Sa{\'a}-Garriga, Albert},
  booktitle={Proceedings of CVPR},
  year={2023}
}
```

---

## Final Notes

- This implementation is a research reproduction and may require further tuning to fully match the paper’s results.
- Contributions and improvements are welcome.
- For issues or questions, please open an issue on this repository.

Happy training and mobile segmentation!