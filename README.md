# CaIRec

CaIRec is the paper-oriented release of our missing-modality recommendation
code. It combines latent modality completion with modality-specific
recommendation projections, modality GCNs, completed-feature item graphs, and
BPR optimization.

## Repository layout

```text
.
├── main.py, model.py, session.py       # training and model implementation
├── dataset_loader.py, evaluation.py    # data protocol and evaluation
├── completion_core/                    # completion runtime
├── configs/                            # retained paper configurations
├── ckpt/                               # fixed Stage 1 projection initializers
├── run_mmrec_mainline.sh                # complete Stage 1.1 → 1.2 → 2 pipeline
├── scripts/                            # Beauty preparation, assets, significance
├── docs/DATASETS.md                    # dataset sources and exact provenance
└── tests/                              # focused regression tests
```

Large local assets are intentionally excluded: `Data/`, `exp_report/`,
`.release_downloads/`, and `.venv/`.

## Environment

Python 3.10 is recommended. Create an isolated environment and install the
recorded dependencies:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch 1.13.0 wheel matching the local CUDA
runtime before installing the remaining requirements if the default wheel is
not suitable.

## Required data

The complete pipeline loads:

- dataset interactions and modality features under `Data/<dataset>/`;
- `unified_missing_items_mr0.5_seed2023.npy` in each dataset directory.

Following I3-MRec, third-party datasets and extracted content features are not
redistributed by this repository:

- **Clothing and Sports:** download the MMRec data from its
  [Google Drive folder](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)
  and place the two directories under `Data/`.
- **Beauty:** download the 2014 Amazon Product Data files
  [`reviews_Beauty_5.json.gz`](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz)
  and
  [`meta_Beauty.json.gz`](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz),
  then run `scripts/prepare_amazon_beauty.py`.

Beauty is derived from the 2014 `Beauty` 5-core data, not the later 2018
`All Beauty` dataset. See [`docs/DATASETS.md`](docs/DATASETS.md) for the exact
raw-file checksums, preprocessing environment, expected output statistics, and
directory layout.

The fixed missing-item payloads are CaIRec-generated files. Install them after
the dataset directories exist:

```bash
python scripts/download_assets.py --payloads all
```

The three small projection-only initializers are committed under
`ckpt/` and mirrored in the
[`v1.0-assets`](https://github.com/ShyJason/CaIRec/releases/tag/v1.0-assets)
release. No pre-existing Stage 1.1, Stage 1.2, or Stage 2 checkpoint is
required.

## Reproduce the complete pipeline

Stage 0 projection training is not part of this repository. Stage 1 starts by
loading the matching pretrained modality projection from
`ckpt/<dataset>.pth`, then the script runs Stage 1.1, Stage
1.2, and Stage 2 in order.

```bash
CHECK_ONLY=1 DATASET=clothing bash run_mmrec_mainline.sh
DATASET=clothing DEVICE_ID=0 bash run_mmrec_mainline.sh
```

```bash
DATASET=beauty DEVICE_ID=1 bash run_mmrec_mainline.sh
DATASET=sports DEVICE_ID=2 bash run_mmrec_mainline.sh
```

Each stage saves its checkpoint under `exp_report/<dataset>/`; the next stage
automatically loads the final checkpoint produced by the preceding stage.
`RUN_TAG` can be set to give all three stages a shared experiment identifier.
An explicit `PROJECTION_CKPT=/path/to/projection.pth` can override the bundled
initializer.

## Tests

With the project environment activated:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```
