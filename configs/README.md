# Paper configurations

Each retained dataset has exactly three canonical configurations:

- `paper_stage1_1.yaml`: estimate the completion model parameters from the
  bundled pretrained projection;
- `paper_stage1_2.yaml`: optimize the completion projection while keeping the
  estimated generative model fixed;
- `paper_stage2.yaml`: train the recommender with the frozen Stage 1.2
  checkpoint.

The three datasets are `clothing`, `beauty`, and `sports`. Stage 2 uses strict
validation-based checkpoint selection and the bundled phase-invariant 50%
missing-modality payload. Dataset-specific early-stopping differences are
written directly in their respective files. All retained Stage 2
configurations use unweighted mean modality fusion.

`run_cairec.sh` is the authoritative entry point. It loads the matching
bundled projection and runs these three configurations in order, passing each
new checkpoint to the next stage.
