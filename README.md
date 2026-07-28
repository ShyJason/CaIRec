# CaIRec

CaIRec is the paper-oriented release of our missing-modality recommendation
code. It combines latent modality completion with modality-specific
recommendation projections, modality GCNs, completed-feature item graphs, and
BPR optimization.

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

## Data

Download Clothing and Sports from the
[MMRec Google Drive folder](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)
and place both directories under `Data/`.

For Beauty, download
[`reviews_Beauty_5.json.gz`](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz)
and
[`meta_Beauty.json.gz`](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz)
into the same directory, then run:

```bash
python3.10 -m venv .venv-beauty
. .venv-beauty/bin/activate
python -m pip install -r requirements-beauty.txt
python scripts/prepare_amazon_beauty.py \
  --source-dir /path/to/beauty-raw \
  --target-dir Data/beauty \
  --device cuda:0
```

Finally, install the fixed missing-item payloads:

```bash
python scripts/download_assets.py --payloads all
```

## Reproduce the complete pipeline

Stage 0 projection training is not part of this repository. Stage 1 starts by
loading the matching pretrained modality projection from
`ckpt/<dataset>.pth`, then the script runs Stage 1.1, Stage
1.2, and Stage 2 in order.

```bash
CHECK_ONLY=1 DATASET=clothing bash run_cairec.sh
DATASET=clothing DEVICE_ID=0 bash run_cairec.sh
```

```bash
DATASET=beauty DEVICE_ID=1 bash run_cairec.sh
DATASET=sports DEVICE_ID=2 bash run_cairec.sh
```

Each stage saves its checkpoint under `exp_report/<dataset>/`; the next stage
automatically loads the final checkpoint produced by the preceding stage.
`RUN_TAG` can be set to give all three stages a shared experiment identifier.
An explicit `PROJECTION_CKPT=/path/to/projection.pth` can override the bundled
initializer.
