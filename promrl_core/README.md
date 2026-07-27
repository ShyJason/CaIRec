# ProMRL Workspace

This directory is a dedicated workspace for `ProMRL` development and
experimentation. It is meant to feel like a focused control room for ProMRL:
the core implementation lives here, the experiment entrypoints are available
here, and the path conventions have been cleaned up so runs can be launched
from here directly.

In practice, this workspace gives us three things:

- a concentrated home for the main `ProMRL` code
- local, editable experiment scripts and configs for ProMRL work
- a lightweight runtime shell that still reuses the repository’s shared data,
  weights, and output directories

## Design Intent

This is not a fully separate project, and it is not a disposable scratch
folder. It is a curated working surface for ProMRL research inside the larger
repository.

The idea is simple:

- keep ProMRL-centric work local to this directory
- avoid bouncing back to the repository root for routine experiments
- preserve access to the existing datasets, checkpoints, and outputs
- reduce accidental edits to the repository-level PMRL scripts and configs

## What Lives Here

### Core Implementation

- `promrl.py`
  Main ProMRL implementation.
- `promrl_3.py`
  An alternative ProMRL branch used for related experiments.
- `promrl_variants.py`
  Variant backends such as `MVAE`, `MoPoE`, `SMIL`, and `Knowledge Bridger`.
- `__init__.py`
  Exposes the local ProMRL modules as a package.

### Workspace Runtime Entry

- `run.py`
  Compatibility launcher for the original ProMRL training CLI. This MMRec
  checkout vendors the ProMRL Python modules used by `model.py`, but it does
  not include the original ProMRL top-level training entrypoint. Set
  `PROMRL_ENTRYPOINT=/path/to/original/run.py` before using the standalone
  ProMRL scripts in this directory.

Example:

```bash
cd promrl_core
PROMRL_ENTRYPOINT=/path/to/original/run.py LOCAL_RANK=0 python ./run.py --help
```

## Local Copies vs Shared Resources

The workspace intentionally mixes local editable files with shared linked
resources.

### Local, Editable Copies

These are real directories inside this workspace and are safe to modify for
ProMRL-specific experiments:

- `config/pmrl`
- `scripts/pmrl`
- `README.md`
- the local ProMRL Python modules in this directory

Edits here do not modify the repository-level originals under the root
`config/pmrl` and `scripts/pmrl`.

### Shared Linked Resources

These are symlinks that expose existing repository or external assets:

- `datasets`
- `MSRVTT`
- `didemo`
- `audiocaps_train`
- `imagenet_1k`
- `pretrained_weights`
- `pretrain_vast`
- `triangle_pretraining`
- `feature_inference_results`
- `outputs`
- `output`
- `results`
- `MM_datasets`
- `SVD_MM`

This means the workspace stays lightweight while still writing results into the
same shared output structure used by the broader project.

## Path Style

Inside this workspace, scripts and configs have been normalized toward
workspace-relative paths. That is the preferred style going forward.

Recommended patterns:

- `./config/pmrl/...`
- `./scripts/pmrl/...`
- `./outputs/...`
- `./datasets/...`
- `./MSRVTT/...`
- `./SVD_MM/...`
- `./pretrain_vast/...`
- `./triangle_pretraining/...`

Avoid reintroducing machine-specific absolute paths.

Keeping paths relative makes this workspace easier to reason about, easier to
move, and less brittle across machines.

## How To Run Experiments From Here

Enter the workspace:

```bash
cd promrl_core
```

Run an existing script:

```bash
export PROMRL_ENTRYPOINT=/path/to/original/run.py
bash scripts/pmrl/pretrain_promrl_msrvtt.sh
```

Or launch manually with the workspace entrypoint:

```bash
PROMRL_ENTRYPOINT=/path/to/original/run.py torchrun ... ./run.py --config ./config/pmrl/...
```

The important convention is that ProMRL experiments should be launched from
this directory if you want the workspace-local script and config copies to be
used.

## Editing Guidance

Good candidates for edits inside this workspace:

- ProMRL model logic
- ProMRL variant logic
- ProMRL-specific configs under `config/pmrl`
- ProMRL experiment scripts under `scripts/pmrl`
- workspace documentation

Things that should generally remain shared rather than duplicated here:

- large datasets
- pretrained weights
- experiment output trees
- generic repository infrastructure unrelated to ProMRL

## Practical Mental Model

If it helps, think of this directory as:

- a ProMRL code capsule
- a ProMRL experiment desk
- a safe place to iterate on ProMRL configs and launch scripts
- a thin shell over the repository’s existing assets

For day-to-day ProMRL work, this should be the default place to read, edit,
launch, and compare experiments.
