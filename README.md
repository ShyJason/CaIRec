# CaIRec

CaIRec is the paper-oriented release of our missing-modality recommendation
code. It combines latent modality completion with modality-specific
recommendation projections, modality GCNs, completed-feature item graphs, and
BPR optimization.

## Main setting

The recorded paper setting uses the `unified_static` missing protocol with the
same pre-generated 50% missing-item payload for training, validation, and test.
The payload seed and model seed are both `2023`.

| Dataset | Recall@20 / NDCG@20 | Best epoch | Fusion reliability |
| --- | ---: | ---: | --- |
| Clothing | 0.08141 / 0.03612 | 280 | fusion, scale 50 |
| Beauty | 0.08418 / 0.03386 | 177 | disabled; mean fusion |
| Sports | 0.10579 / 0.04735 | 239 | graph and fusion, scale 50 |

The shared Stage 2 setting uses a frozen completion module, learning rate
`0.005`, batch size `2048`, strict validation-based checkpoint selection, and
one completed item graph per modality. The graph combines CF and semantic
neighbors with weights `0.4` and `0.6`; the graph residual strength is `0.25`.

## Repository layout

```text
.
├── main.py, model.py, session.py       # training and model implementation
├── dataset_loader.py, evaluation.py    # data protocol and evaluation
├── promrl_core/                        # completion runtime
├── configs/                            # retained paper configurations
├── pretrained_projections/             # fixed Stage 1 projection initializers
├── reproduce_best/20260719/            # fixed Stage 2 reproduction commands
├── scripts/                            # data preparation and significance test
└── tests/                              # focused regression tests
```

Large local assets are intentionally excluded: `Data/`, `exp_report/`,
`.dataset_downloads/`, and `.venv/`.

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

## Required assets

The fixed reproduction commands load:

- dataset interactions and modality features under `Data/<dataset>/`;
- `unified_missing_items_mr0.5_seed2023.npy` in each dataset directory;
- the recorded Stage 1.2 epoch-49 checkpoint under the path declared by each
  reproduction script.

Datasets and Stage 1.2 checkpoints are not committed to Git. The reproduction
scripts fail before training if an expected file or checksum is missing. The
three small projection-only initializers are committed under
`pretrained_projections/` and verified by its `SHA256SUMS`.

## Train the mainline

Stage 0 projection training is not part of this repository. Stage 1 starts by
loading the matching pretrained modality projection from
`pretrained_projections/<dataset>.pth`. The default launch is:

```bash
bash run_mmrec_mainline.sh
```

Set `DATASET=beauty` or `DATASET=sports` for the other retained datasets. An
explicit `PROJECTION_CKPT=/path/to/projection.pth` can override the bundled
initializer. The checkpoint is loaded through the projection-only loader, so
unrelated imputer or recommender tensors are ignored.

## Reproduce the recorded Stage 2 runs

Run the read-only preflight checks first:

```bash
CHECK_ONLY=1 bash reproduce_best/20260719/clothing.sh
CHECK_ONLY=1 bash reproduce_best/20260719/beauty.sh
CHECK_ONLY=1 bash reproduce_best/20260719/sports.sh
```

Then select a physical GPU:

```bash
bash reproduce_best/20260719/clothing.sh 0
bash reproduce_best/20260719/beauty.sh 1
bash reproduce_best/20260719/sports.sh 2
```

These commands reproduce Stage 2 from the retained Stage 1.2 checkpoints; they
do not retrain Stage 1. `SOURCE_SHA256SUMS` fingerprints the source and paper
configuration files in this release. It verifies release integrity, but does
not claim that the historical result-producing source—no longer available—was
recovered. A fresh full run is therefore required before claiming bitwise
reproduction of the recorded metrics.

## Tests

With the project environment activated:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```
