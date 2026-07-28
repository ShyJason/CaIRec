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
├── reproduce_best/20260719/            # fixed Stage 2 reproduction commands
├── scripts/                            # data preparation and significance test
├── tools/                              # projection pretraining
└── tests/                              # focused regression tests
```

Large local assets are intentionally excluded: `Data/`, `exp_report/`,
`.dataset_downloads/`, and `.venv/`.

## Environment

The development environment used Python 3.10 with PyTorch, NumPy, SciPy,
pandas, tqdm, numba, FAISS, EasyDict, and TensorBoard. A pinned public
environment file is still to be added before the archival release.

## Required assets

The fixed reproduction commands load:

- dataset interactions and modality features under `Data/<dataset>/`;
- `unified_missing_items_mr0.5_seed2023.npy` in each dataset directory;
- the recorded Stage 1.2 epoch-49 checkpoint under the path declared by each
  reproduction script.

Datasets and checkpoints are not committed to Git. Their public download
manifest must be supplied separately.

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
do not retrain Stage 1. The historical `SOURCE_SHA256SUMS` records the source
fingerprints used when the reference results were produced. The current source
has subsequently evolved, so a fresh result verification is required before a
versioned archival release.

## Tests

With the project environment activated:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```
