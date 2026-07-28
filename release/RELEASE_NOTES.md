# CaIRec reproducibility assets v1

Tag: `v1.0-assets`

This release contains only CaIRec-generated auxiliary assets required by the
retained Clothing, Beauty, and Sports paper configurations. Third-party
datasets and extracted content features are intentionally not redistributed.

## Assets

| Asset | Installed path |
| --- | --- |
| `cair-missing-payloads-v1.tar.gz` | one fixed payload under each `Data/<dataset>/` |
| `cair-projection-checkpoints-v1.tar.gz` | `ckpt/` |
| `SHA256SUMS` | integrity manifest |

The payload archive contains the unified 50% missing-item payloads generated
with seed 2023. The projection archive mirrors the three small initializers
already committed to the repository.

## Data and installation

Obtain Clothing and Sports from MMRec and build Beauty from the 2014 Amazon
Product Data source as documented in `docs/DATASETS.md`. After the three
dataset directories exist, install the CaIRec payloads:

```bash
python scripts/download_assets.py --payloads all
```

To restore only the projection checkpoint mirror:

```bash
python scripts/download_assets.py --with-checkpoints --overwrite
```

The downloader verifies every archive against the SHA-256 values recorded in
the source tree before extracting it.
