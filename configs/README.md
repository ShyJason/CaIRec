# Config Files

Use `--config` with a YAML or JSON file. CLI args still override config values.

For the supported experiment path, read `docs/MAINLINE.md` first. It is the
source of truth for the current MMRec mainline, canonical configs, traceable
results, and branches that are now retired or ablation-only.

## MMRec Mainline

The MMRec mainline is now the raw-decoder stage-2 recommender with a
multi-source completed-feature post-GCN item-item graph modal residual and the
post-GCN true-missing InfoNCE objective:

```text
PROMRL completion -> raw decoder -> original I3 linear MLP -> modality GCNs -> multi-source completed item-item graph residual -> fusion -> BPR
```

Clothing's current canonical 10% missing-rate mainline is
`configs/clothing/mainline_mr0p1.yaml`. Its key graph knobs are:

- `item_graph_kind=fused_completed`
- `item_graph_topk=10`
- `item_graph_norm=rw`
- `item_graph_cf_weight=0.2`
- `item_graph_image_weight=0.4`
- `item_graph_text_weight=0.4`
- `item_graph_modal_alpha=0.25`
- `item_graph_modal_layers=1`
- `rec_neighbor_cl_weight=0.005`

The default graph is a fused graph built from multiple sources: CF
co-occurrence plus completed-feature modality KNN graphs. Single-source item
graphs are a side branch for ablation. To run one, keep
`item_graph_kind=fused_completed` but set exactly one graph source weight
positive and the others to `0.0`, for example CF-only
`1.0/0.0/0.0` or image-only `0.0/1.0/0.0`.

For every dataset, `configs/<dataset>/stage1_2_decoder_v2.yaml` is the default
best-style no-CL Stage 1.2 setup: `epoch=50`, `alpha_rec=1.0`,
`generative_update_mode=fixed`, `decode_loss_grad_mode=detached`, and
`stage1_2_mode=observed`. Set `missing_rate` per experiment. Stage 1.2 now has
only one target mode: `observed`, which maps both the rec/NLL target and decoder
target to genuinely observed modalities.
The shared Stage 1.2 wrapper `run_stage1_2_baby_imputer_backprop_decoder_v2.sh`
uses the same best-style no-CL defaults for every dataset.
Older Clothing best-style checkpoints may show
`stage1_rec_loss_mode=calmrl_pseudo` in their logs; this is a historical name
used by those checkpoints. New configs and scripts use `stage1_2_mode=observed`.

For all datasets, the default Stage 1.2 upper-bound epoch in
`configs/<dataset>/stage1_2_decoder_v2.yaml` is `50`.

The previous score-level assemble fusion path is retired. Use
`scripts/run_mmrec_mainline_assemble.sh` only with `ALLOW_LEGACY_ASSEMBLE=1`
when intentionally reproducing old results.

Examples:

```bash
cd /path/to/MMRec
python main.py --config configs/clothing/stage1_2_decoder_v2.yaml --imputer_ckpt /path/to/stage1_1_ckpt.pth
python main.py --config configs/clothing/mainline_mr0p1.yaml --imputer_ckpt /path/to/ckpt.pth
```

With the shell wrapper:

```bash
cd /path/to/MMRec
CONFIG=configs/clothing/mainline_mr0p1.yaml \
IMPUTER_CKPT=/path/to/ckpt.pth \
./run_demo_itm.sh
```

Notes:

- Config keys use the same names as CLI arguments, e.g. `train_stage`, `exp_mode`, `lr_decoder`.
- `topk` can be a string or a YAML list.
- Keep checkpoint paths outside the static config if you want to reuse the same config across runs.
