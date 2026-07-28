# Paper configurations

This directory retains only the configurations required by the paper-oriented
CaIRec release.

- `clothing/mainline_mr0p1.yaml`: shared Stage 2 configuration used by the
  fixed Clothing and Sports reproduction scripts. The filename is historical;
  the recorded paper runs use `unified_static` with train/eval missing rate
  `0.5`.
- `clothing/stage1_1_imputer_param.yaml` and
  `clothing/stage1_2_decoder_v2.yaml`: Clothing completion stages.
- `beauty/stage1_1_imputer_param.yaml`,
  `beauty/stage1_2_decoder_v2.yaml`, and `beauty/stage2_decoder_mm.yaml`:
  Beauty completion and recommendation stages.

The fixed commands under `reproduce_best/20260719/` provide the authoritative
runtime overrides for the recorded Stage 2 results.
