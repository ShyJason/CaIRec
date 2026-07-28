# CaIRec reproducibility assets v1

Tag: `v1.0-assets`

This release contains the minimal assets required by the retained Clothing,
Beauty, and Sports paper configurations. Historical feature variants, graph
caches, experiment logs, and result directories are intentionally excluded.

## Assets

| Asset | Installed path |
| --- | --- |
| `cair-data-clothing-v1.tar.gz` | `Data/clothing/` |
| `cair-data-beauty-v1.tar.gz` | `Data/beauty/` |
| `cair-data-sports-v1.tar.gz` | `Data/sports/` |
| `cair-projection-checkpoints-v1.tar.gz` | `projection_checkpoints/` |
| `SHA256SUMS` | integrity manifest |

Each dataset archive contains the interaction table, train/validation/test
splits, original image and text features, user/item mappings, and the unified
50% missing-item payload generated with seed 2023.

## Installation

After cloning CaIRec:

```bash
python scripts/download_assets.py --datasets all
```

The projection checkpoints are already committed in the source repository.
To restore them from this release as well:

```bash
python scripts/download_assets.py --datasets all --with-checkpoints --overwrite
```

The downloader verifies every archive against the SHA-256 values recorded in
the source tree before extracting it.
