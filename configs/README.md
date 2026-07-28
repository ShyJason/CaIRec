# Paper configurations

Each retained dataset has exactly three canonical configurations:

- `paper_stage1_1.yaml`: estimate the completion model parameters from the
  bundled pretrained projection;
- `paper_stage1_2.yaml`: optimize the completion projection while keeping the
  estimated generative model fixed;
- `paper_stage2.yaml`: train the recommender with the frozen Stage 1.2
  checkpoint.

The three datasets are `clothing`, `beauty`, and `sports`. Stage 2 uses the
strict validation-selection protocol and the fixed `unified_static` 50%
missing-modality payload. Dataset-specific seed and early-stopping differences
are written directly in their respective files. All retained Stage 2
configurations use unweighted mean modality fusion; posterior reliability and
the Rec Neighbor contrastive loss are disabled.

The commands under `reproduce_best/20260719/` are the authoritative entry
points for the current Stage 2 setup. The directory name is retained for
compatibility with existing asset paths.
