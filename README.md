# DS-TSNet: Underwater Image Classification

Modular research code for the DS-TSNet workflow: DSACM/TWSSL preprocessing, DenseNet feature extraction, statistical pre-filtering, Binary Weighted Grasshopper Optimization (B-WGOA), and MLP classification. The repository also includes runnable CNN baselines and feature-level ablation/baseline scripts.

> **Reproducibility note.** This repository is an organized research implementation based on the supplied script and manuscript. It has been syntax-checked, but full training has not been executed end-to-end in this environment. Verify every reported metric by rerunning experiments with the exact dataset split, class mapping, preprocessing, checkpoint selection, and software versions used for the paper.

## Repository structure

- `config.py`: seed, paths, model, filter, WGOA, and MLP configuration.
- `src/data.py`: deterministic ImageFolder loading.
- `src/backbones.py`: frozen DenseNet201/121 feature extractors.
- `src/feature_extraction.py`: hybrid feature-bank construction.
- `src/clahe.py`, `src/dsacm.py`, `src/twssl.py`: preprocessing components.
- `src/prefilter.py`, `src/fitness.py`, `src/wgoa.py`: filtering, fitness, B-WGOA.
- `src/mlp.py`, `src/evaluation.py`: MLP and metrics.
- `experiments/run_cnn_baselines.py`: EfficientNet-B3, ResNet-50, ResNet-101 image-level baselines.
- `experiments/run_feature_baselines.py`: classical classifiers and an RFE feature-selection baseline.
- `experiments/run_feature_ablations.py`: feature-filtering/selection ablations.
- `experiments/manuscript_reference_results.csv`: manuscript-reported reference metrics for audit only, not new experimental results.

## Setup

Python 3.10 or 3.11 is recommended. Install dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA, install the PyTorch/torchvision builds appropriate for your hardware from the official PyTorch installation selector, then install the remaining packages.

## Dataset layout

The main pipeline expects ImageFolder directories with identical class subfolders in the train and evaluation roots, e.g.:

```text
dataset/
  train/
    Echinus/ ...
    Holothurian/ ...
    Scallop/ ...
    Starfish/ ...
  test/
    Echinus/ ...
    Holothurian/ ...
    Scallop/ ...
    Starfish/ ...
```

Do not commit the dataset or trained weights to GitHub unless you have permission to redistribute them. Update `TRAIN_ROOT`, `VAL_ROOT`, and `OUT_DIR` in `config.py` for your machine.

## Run proposed pipeline

```bash
pip install -r requirements.txt
python main.py
```

The provided configuration preserves the modularized script's values: seed 42, input size 224, batch size 32, hybrid feature dimension 2,944, variance→MI prefilter to 1,200 dimensions, B-WGOA population 32 / up to 10 iterations, and MLP hidden layers 1024/512. The manuscript also reports 128×128 in its numerical configuration; reconcile the final experimental resolution with the source run before reporting a reproduction. The code keeps the input script's configured `IMG_SIZE=224` rather than silently changing it.

## Baselines and ablations

See [`experiments/README.md`](experiments/README.md). Typical commands:

```bash
python -m experiments.run_cnn_baselines --train-root /path/train --val-root /path/val --test-root /path/test
python -m experiments.run_feature_baselines --features-dir /kaggle/working/hybrid_wgoa_mlp_out
python -m experiments.run_feature_ablations --features-dir /kaggle/working/hybrid_wgoa_mlp_out
```

## Important scientific implementation note

The supplied feature extraction code routes labels `{0, 1, 3}` to DenseNet201 and label `{2}` to DenseNet121, zeroing the other bank per sample. This behavior is preserved to avoid silently changing the source script, but it is not conventional per-image concatenation of both feature vectors. Confirm this is intended before interpreting results. Also, use a dedicated validation split for all model/threshold/epoch selection and evaluate the external test split only once.

## Outputs

Main pipeline outputs are written to `OUT_DIR`; experiment scripts save JSON/CSV metrics and checkpoints under their output directories. Keep exact package versions, dataset split manifests, class mappings, seeds, and commit hashes with each run.

## License

Add the license appropriate to your institution/project before making the repository public. Do not imply that third-party datasets or pretrained weights are relicensed by this repository.
