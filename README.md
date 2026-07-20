# ForensicConcept: Transferable Forensic Concepts for AIGI Detection

Official PyTorch implementation of **ForensicConcept**, accepted at the 43rd
International Conference on Machine Learning (ICML 2026).

[[arXiv](https://arxiv.org/abs/2606.07034)]

ForensicConcept turns diffuse forensic cues into explicit, reusable concepts.
It first discovers discriminative concepts with a strong forensic encoder, then
transfers the resulting concept codebook to other detector backbones.

![Overview of forensic concept learning](assets/readme/method_overview.png)

## Method

The framework has three stages:

1. **Adapter-guided discriminative tuning (ADT)** inserts LoRA adapters into a
   frozen DINOv3 encoder and trains a binary real/fake classifier.
2. **Unsupervised concept induction (UCI)** localizes high-evidence patches with
   Transformer attribution and clusters their features into a forensic concept
   codebook.
3. **Concept-aligned projection (CAP)** maps the global image representation to
   the induced concept space, producing an interpretable concept prediction in
   addition to the standard classifier output.

For cross-backbone transfer, **concept-guided codebook injection (CGCI)** scores
patch-to-concept similarity, selects the most relevant evidence, and aggregates
it into a concept-based prediction. CleanDIFT serves as a generation-trace
reference, while CKNNA measures whether another backbone preserves compatible
local neighborhoods.

![Concept-guided codebook injection](assets/readme/concept_injection.png)

## Results

The following image-level accuracies are reported in the paper. All three main
benchmarks use a detector trained on Stable Diffusion 1.4 images.

| Evaluation benchmark | DINOv3 without concepts | ForensicConcept |
|---|---:|---:|
| GenImage, mean over 8 generators | 90.7 | **92.0** |
| GAN-family benchmark, mean over 7 generators | 87.3 | **90.1** |
| Chameleon | 83.7 | **84.4** |
| Mean over the three benchmarks | 87.2 | **88.8** |

The learned codebook also transfers to CLIP ViT-L/14: CGCI raises its GenImage
mean accuracy from **83.7** to **88.2** (+4.5 points). The visualization below
shows how codebook injection redirects attention toward more meaningful
forensic regions.

![Model-attended patches before and after codebook injection](assets/readme/evidence_comparison.png)

## Released implementation

This release contains the two detector families used by the method:

- **DINOv3 ViT-L/16** with LoRA and a trainable concept head.
- **CLIP ViT-L/14** with visual LoRA and a patch-token codebook head.

The implementation is intentionally limited to the reproducible core. Experiment
logs, private datasets, concept-extraction notebooks, ablation-only detectors,
and third-party DINOv3 source code are not included. Every public configuration
uses a `224 x 224` input.

## Repository layout

```text
.
|-- assets/
|   |-- codebooks/          # Released CLIP codebook
|   `-- readme/             # README figures exported from the paper
|-- configs/                # Stage-1 and stage-2 configurations
|-- data/                   # Dataset discovery and preprocessing
|-- models/                 # CLIP backbone and codebook head
|-- networks/               # DINOv3 detector and training wrapper
|-- options/                # Shared runtime options
|-- plugins/                # Optional DINOv3 distributed acceleration
|-- train_with_config.py    # Training entry point
|-- test_with_config.py     # Evaluation entry point
`-- validate.py             # ACC, AUC, and AP evaluation
```

## Installation

Python 3.10 or newer is required. Confirm that `python3 --version` points to the intended environment, install PyTorch for the CUDA version available on your machine, then install the remaining dependencies:

```bash
python3 -m pip install -r requirements.txt
```

DINOv3 is an external dependency. Clone the [official repository](https://github.com/facebookresearch/dinov3) into the default relative location:

```bash
git clone https://github.com/facebookresearch/dinov3.git ./external/dinov3
```

The integration was tested with DINOv3 package version `0.0.1`. You may use another clone location by changing `training.dinov3_repo_path`.

## Weights

The detector checkpoints and pretrained backbones are released separately. Place them as follows:

```text
weights/
|-- backbones/
|   |-- ViT-L-14.pt
|   `-- dinov3-vitl16/
|       `-- model.safetensors
|-- concepts/
|   `-- dinov3_concept_matrix.npy       # 200 x 1024
`-- detectors/
    |-- dinov3_vitl16_lora_stage1.pth
    |-- dinov3_vitl16_concept_stage2.pth
    |-- clip_vitl14_lora_stage1.pth
    `-- clip_vitl14_codebook_stage2.pth
```

The CLIP codebook is already included:

```text
assets/codebooks/cleandift_codebook.npy  # 200 x 1280, float32
```

It contains the raw cluster centers. `clip_codebook_l2: true` normalizes them when the model is constructed. The stage-2 CLIP checkpoint also stores the normalized matrix in `codebook_head.codebook`.

## Data layout

Images are discovered recursively. Use one of the supported binary class layouts, preferably `0_real/1_fake`:

```text
datasets/
|-- train/
|   |-- 0_real/
|   `-- 1_fake/
|-- val/
|   |-- 0_real/
|   `-- 1_fake/
`-- test/
    `-- ADM/
        |-- 0_real/
        `-- 1_fake/
```

The loader also recognizes `real/fake`, `raw/synthesis`, and `0_fake/1_real`. Update `data.*` and `testing.groups` in the selected YAML file when using another layout.

## Evaluation

All commands should be run from the repository root. Evaluate the released DINOv3 detector:

```bash
python3 test_with_config.py \
  --config configs/dinov3_concept.yaml \
  --checkpoint_path weights/detectors/dinov3_vitl16_concept_stage2.pth
```

Evaluate the released CLIP-codebook detector:

```bash
python3 test_with_config.py \
  --config configs/clip_codebook.yaml \
  --checkpoint_path weights/detectors/clip_vitl14_codebook_stage2.pth
```

Evaluation reports accuracy, ROC-AUC, and average precision for each dataset and their mean. DINOv3 reports both the main and concept heads, including configured concept sparsity ratios. CLIP reports the main classifier and `cb_only` codebook branch.

Evaluation exits with an error when none of the configured dataset paths can be evaluated. Replace the placeholder `testing.groups` paths in the selected YAML before running.

`clip_cb_tau_w` is a runtime codebook-pooling temperature. It is not stored in the checkpoint and may be tuned during inference; the public configuration keeps the training default `0.05`.

## Training

Both model families use two stages. Stage 2 initializes from the corresponding stage-1 checkpoint through `training.epoch`.

### DINOv3

```bash
# Stage 1: visual LoRA + classifier
python3 train_with_config.py --config configs/dinov3_lora.yaml

# Stage 2: concept mapping + concept classifier
python3 train_with_config.py --config configs/dinov3_concept.yaml
```

Before stage 2, place or rename the selected stage-1 checkpoint to:

```text
weights/detectors/dinov3_vitl16_lora_stage1.pth
```

### CLIP

```bash
# Stage 1: visual qkv LoRA + classifier
python3 train_with_config.py --config configs/clip_lora.yaml

# Stage 2: codebook projection + classifier
python3 train_with_config.py --config configs/clip_codebook.yaml
```

Before stage 2, place or rename the selected stage-1 checkpoint to:

```text
weights/detectors/clip_vitl14_lora_stage1.pth
```

Checkpoints and an immutable copy of the effective configuration are written under `models.checkpoints_dir/training.name`.

## Checkpoint behavior

- DINOv3 stage-2 checkpoints contain the classifier, concept mapping/head, and all LoRA tensors. A separate stage-1 checkpoint is not required for stage-2 inference.
- CLIP stage-2 checkpoints contain visual LoRA, the main classifier, and the complete codebook head.
- Explicit checkpoint files are accepted through `--checkpoint_path`.
- DINOv3 LoRA loading verifies every adapter key, shape, and tensor value before evaluation.

## External code and models

DINOv3 source and pretrained weights are distributed by Meta under their own terms. CLIP pretrained weights are distributed by OpenAI under their own terms. Review the corresponding upstream licenses and model cards before use or redistribution.

Third-party source notices included with this release are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

The original code in this repository is released under the
[MIT License](LICENSE). Third-party components and external model weights
remain subject to their respective licenses and terms.

## Smoke tests

After installing the dependencies, run the release checks with:

```bash
python3 -m unittest discover -s tests -v
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhou2026forensicconcept,
  title     = {{ForensicConcept}: Transferable Forensic Concepts for {AIGI} Detection},
  author    = {Zhou, Menyanshu and Zhou, Ziyin and Sun, Ke and Luo, Yunpeng and Ji, Jiayi and Sun, Xiaoshuai and Ji, Rongrong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
