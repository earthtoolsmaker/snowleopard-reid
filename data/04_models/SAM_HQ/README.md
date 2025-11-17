# SAM HQ Model Checkpoints

This directory contains SAM HQ (Segment Anything Model - High Quality) model checkpoints.

## Download Instructions

To download the SAM HQ vit_b checkpoint (~379MB):

```bash
# From the project root directory
wget -P data/04_models/SAM_HQ \
  https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth
```

## Available Model Variants

### ViT-Base (Recommended)
- **File**: `sam_hq_vit_b.pth`
- **Size**: ~379MB
- **Download**:
  ```bash
  wget -P data/04_models/SAM_HQ https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth
  ```
- **Speed**: Fastest
- **Quality**: Good

### ViT-Large
- **File**: `sam_hq_vit_l.pth`
- **Size**: ~1.2GB
- **Download**:
  ```bash
  wget -P data/04_models/SAM_HQ https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth
  ```
- **Speed**: Medium
- **Quality**: Better

### ViT-Huge
- **File**: `sam_hq_vit_h.pth`
- **Size**: ~2.4GB
- **Download**:
  ```bash
  wget -P data/04_models/SAM_HQ https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
  ```
- **Speed**: Slowest
- **Quality**: Best

## Usage

The model checkpoint is used by the SAM HQ segmentation script:

```bash
uv run python scripts/models/sam_hq.py \
  --input-dir ./data/02_processed/locations \
  --predictions-dir ./data/05_model_output/grounding_dino/predictions \
  --output-dir ./data/05_model_output/sam_hq \
  --checkpoint ./data/04_models/SAM_HQ/sam_hq_vit_b.pth \
  --model-type vit_b
```

## More Information

- **SAM HQ Paper**: https://arxiv.org/abs/2306.01567
- **SAM HQ GitHub**: https://github.com/SysCV/sam-hq
- **HuggingFace Models**: https://huggingface.co/lkeab/hq-sam
