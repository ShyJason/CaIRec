# Clothing Configs

Use `mainline_mr0p1.yaml` for the current verified Clothing 10% missing-rate
mainline.

Current canonical setting:

```text
stage1.2 style = best-style no-CL
item_graph_topk = 10
item_graph_cf_weight = 0.2
item_graph_image_weight = 0.4
item_graph_text_weight = 0.4
item_graph_modal_alpha = 0.25
rec_neighbor_cl_weight = 0.005
```

For Clothing Stage 1.2, `stage1_2_decoder_v2.yaml` is the default imputer
config. It uses the best-style no-CL branch with `stage1_2_mode=observed`
(observed recommendation guidance, observed decode target, detached decode loss,
fixed generative update).
Historical best-style Clothing checkpoints may log this branch as
`calmrl_pseudo`; that name is historical. New runs should use
`stage1_2_mode=observed`.

The older `stage2_*` files are kept for historical runs, ablations, and
reproduction of previous command paths. They should not be treated as the
current Clothing mainline unless `docs/MAINLINE.md` explicitly says so.
