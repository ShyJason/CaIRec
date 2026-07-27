# VAE Completion Variant

This directory is an isolated VAE-based modality completion prototype. It does
not patch the main MMRec/CalMRL implementation.

## What It Implements

The completion model is a conditional VAE over item modalities:

```text
q_phi(beta | x_obs, mask) -> mu, logvar
beta = mu + eps * std
p_theta(x_v | beta), p_theta(x_t | beta)
```

Training uses only observed modalities as reconstruction targets. The base
loss is:

```text
loss = observed_reconstruction_mse + beta_kl * KL(q_phi(beta | x_obs, mask) || N(0, I))
```

Official missing modalities are never used as training targets. Evaluation
compares the generated missing modality against the full feature table only for
reporting imputation quality.

The current optimizer also supports five completion-side enhancements:

- `--input_dropout`: randomly hides a fraction of observed modalities at the
  VAE input while keeping the official observed modalities as reconstruction
  targets. This is the strongest simple improvement so far because it trains
  single-view completion directly.
- `--latent_consistency_weight`: for items with both modalities observed, align
  the image-only and text-only posterior means to the full-view posterior mean.
  Use `--latent_consistency_start_epoch` to delay this loss; starting it at
  epoch 1 was unstable in Clothing 30%.
- `--single_view_rec_weight`: for items with both modalities observed, force
  image-only and text-only inputs to reconstruct both modalities. This directly
  trains cross-modal completion and is the strongest completion-quality
  improvement so far.
- `--fusion_consistency_weight`: for items with both modalities observed,
  complete image-only and text-only views, then align those completed
  item-level fusion embeddings with the real full-modal fusion embedding. This
  is a completion-side self-distillation target that is closer to the
  recommendation representation space than raw feature MSE.
- `--cf_align_weight`: aligns lightweight modality heads using CF neighbor
  positives sampled from the user-item graph. This is a recommendation-space
  regularizer, not an oracle completion target.

With all optional terms enabled, the loss is:

```text
loss =
  observed_reconstruction_mse
  + beta_kl * KL(q_phi(beta | x_obs, mask) || N(0, I))
  + latent_consistency_weight * latent_consistency
  + single_view_rec_weight * single_view_reconstruction
  + fusion_consistency_weight * completed_fusion_consistency
  + cf_align_weight * cf_alignment
```

## Files

- `train_vae_imputer.py`: train/evaluate/export the VAE imputer.
- `run_clothing_mr0p3.sh`: default Clothing 30% missing launch script.

## Output Layout

The exporter writes:

```text
<output_dir>/
  ckpt_best.pt
  ckpt_last.pt
  metrics.json
  manifest.json
  phase_train/
    image_feat.npy
    text_feat.npy
    completed_image_feat.npy
    completed_text_feat.npy
    posterior_mu.npy
    posterior_logvar.npy
  phase_eval/
    image_feat.npy
    text_feat.npy
    completed_image_feat.npy
    completed_text_feat.npy
    posterior_mu.npy
    posterior_logvar.npy
  phase_graph/
    image_feat.npy
    text_feat.npy
    image_observed_mask.npy
    text_observed_mask.npy
```

`phase_train/image_feat.npy` and `phase_train/text_feat.npy` are duplicated
from the completed files so they can be used by MMRec's
`modal_feature_override_dir` path.

`phase_graph` is for item-item graph construction. It treats any item modality
that is missing in train, validation, or test as missing, then fills it with the
VAE output. This avoids building semantic graph edges from held-out oracle
features.

## Default Clothing Run

```bash
DEVICE_ID=0 variants/vae_completion/run_clothing_mr0p3.sh
```

The script uses:

- dataset: `clothing`
- train missing rate: `0.3`
- eval missing rate: `0.5`
- seed: `2023`
- normalized raw features
- observed-only reconstruction

## Clothing 30% Missing Notes

All rows below use Clothing train missing rate `0.3`, eval missing rate `0.5`,
seed `2023`, normalized raw features, and the same basic MMRec stage2 ii-graph
setting with frozen exported VAE features.

| VAE features | completion test cosine | completion test MSE | stage2 Recall@20 | stage2 NDCG@20 | feature dir |
| --- | ---: | ---: | ---: | ---: | --- |
| base VAE | 0.660470 | 0.000929 | 0.07394 | 0.03347 | `exp_report/clothing/vae_completion_clothing_mr0p3_iigraph_full_20260703_141523/vae_features` |
| `input_dropout=0.8` | 0.710481 | 0.000678 | 0.07447 | 0.03360 | `exp_report/clothing/vae_completion_imputer_enhance_20260703_212022/vae_inputdrop08/vae_features` |
| `input_dropout=0.8`, `cf_align_weight=3e-5` | 0.710231 | 0.000675 | 0.07510 | 0.03371 | `exp_report/clothing/vae_completion_recspace_20260704_014030/vae_inputdrop08_cf3e5/vae_features` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80` | 0.713497 | 0.000665 | 0.07536 | 0.03410 | `exp_report/clothing/vae_completion_latent_consistency_20260704_0200/vae_inputdrop08_lc1e3_s80/vae_features` |
| `input_dropout=0.8`, `cf_align_weight=3e-5`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80` | 0.711864 | 0.000668 | 0.07494 | 0.03373 | `exp_report/clothing/vae_completion_combo_20260704_0238/vae_inputdrop08_cf3e5_lc1e3_s80/vae_features` |
| `input_dropout=0.8`, `cf_align_weight=1e-5`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80` | 0.711779 | 0.000670 | 0.07455 | 0.03350 | `exp_report/clothing/vae_completion_cfweak_20260704_weakcf_lc/vae_inputdrop08_cf1e5_lc1e3_s80/vae_features` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80`, `single_view_rec_weight=0.5` | 0.731825 | 0.000606 | 0.07550 | 0.03423 | `exp_report/clothing/vae_completion_singleview_20260704/vae_inputdrop08_lc1e3_s80_sv05/vae_features` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80`, `single_view_rec_weight=0.875` | 0.739545 | 0.000584 | 0.07620 | 0.03442 | `exp_report/clothing/vae_completion_singleview_20260704/vae_inputdrop08_lc1e3_s80_sv0875/vae_features` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80`, `single_view_rec_weight=1.0` | 0.742164 | 0.000577 | 0.07635 | 0.03451 | `exp_report/clothing/vae_completion_singleview_20260704/vae_inputdrop08_lc1e3_s80_sv10/vae_features` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80`, `single_view_rec_weight=1.0`, `fusion_consistency_weight=1e-3` | 0.727055 | 0.000586 | 0.07364 | 0.03326 | `exp_report/clothing/vae_completion_fusion_consistency_20260704/vae_inputdrop08_lc1e3_s80_sv10_fc1e3` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80`, `single_view_rec_weight=1.125` | 0.743537 | 0.000574 | 0.07603 | 0.03400 | `exp_report/clothing/vae_completion_singleview_20260704/vae_inputdrop08_lc1e3_s80_sv1125/vae_features` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80`, `single_view_rec_weight=1.25` | 0.745325 | 0.000569 | 0.07604 | 0.03443 | `exp_report/clothing/vae_completion_singleview_20260704/vae_inputdrop08_lc1e3_s80_sv125/vae_features` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80`, `single_view_rec_weight=1.5` | 0.748705 | 0.000558 | 0.07544 | 0.03382 | `exp_report/clothing/vae_completion_singleview_20260704/vae_inputdrop08_lc1e3_s80_sv15/vae_features` |
| `input_dropout=0.8`, `latent_consistency_weight=1e-3`, `latent_consistency_start_epoch=80`, `single_view_rec_weight=2.0` | 0.755204 | 0.000538 | 0.07580 | 0.03411 | `exp_report/clothing/vae_completion_singleview_20260704/vae_inputdrop08_lc1e3_s80_sv20/vae_features` |

Current conclusion:

- Best pure completion quality: `input_dropout=0.8` plus delayed latent
  consistency (`1e-3`, start epoch `80`) and single-view reconstruction
  (`2.0`).
- Best final strict Recall@20/NDCG@20 among these basic stage2 runs:
  `single_view_rec_weight=1.0`. Neighboring values `0.875` and `1.125` were
  also tested and did not improve Recall@20/NDCG@20. Increasing this weight to
  `1.25`, `1.5`, or `2.0` further improves imputation cosine/MSE, but does not
  improve recommendation transfer in the frozen-feature stage2 setting.
- `fusion_consistency_weight=1e-3` is too strong in the current form. It lowers
  raw missing-modality completion quality and also lowers downstream Recall@20
  from `0.07635` to `0.07364`. If this idea is revisited, use a much smaller
  weight such as `1e-4` or start it after the base single-view imputer has
  already learned stable reconstruction.
- CF alignment can improve recommendation transfer over pure input dropout, but
  combined with latent consistency it is not additive on final strict Recall@20.
  Lowering it from `3e-5` to `1e-5` also did not help. Do not treat CF alignment
  as the main VAE completion improvement path unless the loss is redesigned.

## Notes

- The saved `*_mask_mr*.npy` files use `True = missing`. Internally this
  variant converts them to `observed = ~missing`.
- The exported `image_observed_mask.npy` / `text_observed_mask.npy` files use
  `True = observed` and are consumed by MMRec's
  `item_graph_feature_source=external_completed` path.
- The model is intentionally separate from CalMRL. There are no CalMRL
  `W/mu/sigma`, no closed-form posterior, and no stage1.1/stage1.2 EM update.
- The best current completion setting keeps the default observed-only target
  protocol, trains the input as denoising/single-view completion with
  `--input_dropout 0.8`, delays latent consistency to epoch `80`, and adds
  explicit single-view reconstruction from each single modality to both
  modalities with `--single_view_rec_weight 2.0`.
- The best current recommendation setting uses the same recipe but with
  `--single_view_rec_weight 1.0`, which is a better balance between raw
  completion fidelity and downstream ranking utility.
