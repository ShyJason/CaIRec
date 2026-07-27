#!/usr/bin/env python3
"""Sequential Sports stage2 hyperparameter search.

Searches only the five agreed hyperparameters, using seed=1 and strict val
Recall@20 selection, then runs five-seed verification for the selected config,
the no-CL counterpart, and the current mainline config.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = "sports"
BASELINE_TAG = "mmrec_sports_mm_fixedmissing_20260524_165817"
SEARCH_SEED = 1
FULL_SEEDS = [1, 12, 123, 1234, 12345]
TIE_EPS = 0.0003


@dataclasses.dataclass(frozen=True)
class Config:
    batch_size: int = 256
    lr_rec: float = 0.001
    modality_bpr_coeff: float = 0.2
    reg_coeff: float = 1e-4
    rec_neighbor_cl_weight: float = 0.01
    rec_neighbor_cl_temp: float = 0.25
    rec_neighbor_cl_bank_size: int = 256

    def with_update(self, **kwargs: object) -> "Config":
        return dataclasses.replace(self, **kwargs)

    def as_cli(self) -> list[str]:
        return [
            "--batch_size",
            str(self.batch_size),
            "--lr",
            fmt_float(self.lr_rec),
            "--lr_rec",
            fmt_float(self.lr_rec),
            "--modality_bpr_coeff",
            fmt_float(self.modality_bpr_coeff),
            "--reg_coeff",
            fmt_float(self.reg_coeff),
            "--rec_neighbor_cl_weight",
            fmt_float(self.rec_neighbor_cl_weight),
            "--rec_neighbor_cl_temp",
            fmt_float(self.rec_neighbor_cl_temp),
            "--rec_neighbor_cl_bank_size",
            str(self.rec_neighbor_cl_bank_size),
        ]


@dataclasses.dataclass
class RunResult:
    candidate: str
    seed: int
    log_path: pathlib.Path
    best_epoch: int | None
    val_recall20: float | None
    val_ndcg20: float | None
    test_recall20: float | None
    test_ndcg20: float | None
    status: str


def fmt_float(value: float) -> str:
    return f"{value:.10g}"


def slug_float(value: float) -> str:
    text = fmt_float(value).replace("-", "m").replace(".", "p")
    return text.replace("+", "").replace("e", "e")


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[sports-hparam] {stamp} {message}", flush=True)


def seed_tag(seed: int) -> str:
    return f"mmrec_sports_mm_mr0.3_seed{seed}_{BASELINE_TAG}"


def find_best_ckpt_from_log(log_path: pathlib.Path, stage_dir: pathlib.Path) -> pathlib.Path:
    if not log_path.exists():
        raise FileNotFoundError(f"missing stage1_2 log: {log_path}")
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"best epoch\s+(\d+)", text)
    if not matches:
        raise RuntimeError(f"no best epoch in {log_path}")
    epoch = int(matches[-1])
    ckpts = sorted((stage_dir / "ckpt").glob(f"*_epoch{epoch}.pth"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint for epoch {epoch} under {stage_dir / 'ckpt'}")
    return ckpts[-1]


def ckpt_for_seed(seed: int) -> pathlib.Path:
    override = os.environ.get(f"IMPUTER_CKPT_SEED_{seed}") or os.environ.get("IMPUTER_CKPT")
    if override:
        ckpt = pathlib.Path(override).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint override does not exist: {ckpt}")
        return ckpt

    tag = seed_tag(seed)
    report_log = ROOT / "exp_report" / DATASET / "pipeline_reports" / f"{tag}_raw_decoder_mm" / "stage1_2.log"
    stage_dir = ROOT / "exp_report" / DATASET / f"stage1_2_sports_beststyle_nocl_{tag}"
    return find_best_ckpt_from_log(report_log, stage_dir)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_log(candidate: str, seed: int, log_path: pathlib.Path) -> RunResult:
    if not log_path.exists():
        return RunResult(candidate, seed, log_path, None, None, None, None, None, "missing_log")

    text = strip_ansi(log_path.read_text(errors="ignore"))
    bests = re.findall(r"best epoch\s+(\d+)", text)
    if not bests:
        return RunResult(candidate, seed, log_path, None, None, None, None, None, "missing_best")

    best_epoch = int(bests[-1])
    epoch_pattern = re.compile(
        rf"epoch\s*=\s*{best_epoch}\s+hr@20\s*=\s*([0-9.]+),\s*"
        rf"recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)"
    )
    epoch_matches = epoch_pattern.findall(text)
    if not epoch_matches:
        return RunResult(candidate, seed, log_path, best_epoch, None, None, None, None, "missing_val_metric")

    _, val_recall, val_ndcg = epoch_matches[-1]
    final_matches = re.findall(
        r"final strict test hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)",
        text,
    )
    test_recall = test_ndcg = None
    if final_matches:
        _, test_recall_text, test_ndcg_text = final_matches[-1]
        test_recall = float(test_recall_text)
        test_ndcg = float(test_ndcg_text)

    return RunResult(
        candidate=candidate,
        seed=seed,
        log_path=log_path,
        best_epoch=best_epoch,
        val_recall20=float(val_recall),
        val_ndcg20=float(val_ndcg),
        test_recall20=test_recall,
        test_ndcg20=test_ndcg,
        status="ok",
    )


def has_completed(log_path: pathlib.Path) -> bool:
    return log_path.exists() and "best epoch" in log_path.read_text(errors="ignore")


def run_one(
    *,
    phase_dir: pathlib.Path,
    candidate: str,
    seed: int,
    config: Config,
    run_tag: str,
    device_id: str,
    dry_run: bool,
    epochs: int,
    early_stop: int,
    strict_probe_test_interval: int,
) -> RunResult:
    out_dir = phase_dir / candidate
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"seed{seed}.log"
    suffix = f"stage2_sports_hparam_{candidate}_dseed0_seed{seed}_{run_tag}"

    if has_completed(log_path):
        log(f"skip existing candidate={candidate} seed={seed}")
        return parse_log(candidate, seed, log_path)

    ckpt = ckpt_for_seed(seed)
    cmd = [
        str(ROOT / "run_stage2_baby_recommender_decoder.sh"),
        "--seed",
        str(seed),
        "--dataset_seed",
        "0",
        "--missing_mask_protocol",
        "i3",
        "--selection_mode",
        "val",
        "--evaluation_protocol",
        "strict",
        "--recommendation_selection_metric",
        "recall",
        "--recommendation_selection_topk",
        "20",
        "--topk",
        "[20]",
        "--eva_interval",
        "1",
        "--log",
        "1",
        *config.as_cli(),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CONFIG": "configs/sports/stage2_decoder_mm.yaml",
            "DATASET": DATASET,
            "EXP_MODE": "mm",
            "DATASET_SEED": "0",
            "DEVICE_ID": device_id,
            "USE_GPU": "1",
            "TENSORBOARD": "0",
            "SAVE": "1",
            "IMPUTER_CKPT": str(ckpt),
            "SUFFIX": suffix,
            "EPOCHS": str(epochs),
            "EARLY_STOP": str(early_stop),
            "EVA_INTERVAL": "1",
            "STRICT_PROBE_TEST_INTERVAL": str(strict_probe_test_interval),
            "LR_IMP": "0.0002",
            "LR_DECODER": "0.00005",
            "PYTHONUNBUFFERED": "1",
            "MKL_THREADING_LAYER": env.get("MKL_THREADING_LAYER", "GNU"),
        }
    )

    log(f"run candidate={candidate} seed={seed} bs={config.batch_size} lr={config.lr_rec}")
    with log_path.open("w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.write(f"# IMPUTER_CKPT={ckpt}\n")
        if dry_run:
            f.write("DRY_RUN=1\n")
            f.write(f"CONFIG={config}\n")
            return RunResult(candidate, seed, log_path, None, None, None, None, None, "dry_run")
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return parse_log(candidate, seed, log_path)


def write_rows(path: pathlib.Path, rows: Iterable[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("candidate\tseed\tbest_epoch\tval_R20\tval_N20\ttest_R20\ttest_N20\tstatus\tlog_path\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        row.candidate,
                        str(row.seed),
                        "" if row.best_epoch is None else str(row.best_epoch),
                        "" if row.val_recall20 is None else f"{row.val_recall20:.5f}",
                        "" if row.val_ndcg20 is None else f"{row.val_ndcg20:.5f}",
                        "" if row.test_recall20 is None else f"{row.test_recall20:.5f}",
                        "" if row.test_ndcg20 is None else f"{row.test_ndcg20:.5f}",
                        row.status,
                        str(row.log_path),
                    ]
                )
                + "\n"
            )


def choose_best(step_name: str, rows: list[RunResult], prefer_larger_batch_on_tie: bool = False) -> str:
    ok = [row for row in rows if row.status == "ok" and row.val_recall20 is not None]
    if not ok:
        raise RuntimeError(f"no completed rows for {step_name}")
    ok.sort(key=lambda row: (row.val_recall20 or -1.0, row.val_ndcg20 or -1.0), reverse=True)
    best = ok[0]
    if prefer_larger_batch_on_tie:
        near = [row for row in ok if best.val_recall20 is not None and row.val_recall20 is not None and best.val_recall20 - row.val_recall20 < TIE_EPS]
        near.sort(key=lambda row: int(row.candidate.removeprefix("bs")), reverse=True)
        best = near[0]
    return best.candidate


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def write_final_summary(path: pathlib.Path, rows: list[RunResult]) -> None:
    grouped: dict[str, list[RunResult]] = {}
    for row in rows:
        if row.status == "ok":
            grouped.setdefault(row.candidate, []).append(row)

    with path.open("w") as f:
        f.write(
            "candidate\tok_count\tRecall20_mean\tRecall20_std\tNDCG20_mean\tNDCG20_std\t"
            "Recall20_drop_min_mean\tNDCG20_drop_min_mean\tseed_scores\n"
        )
        for candidate in sorted(grouped):
            vals = grouped[candidate]
            recall = [row.test_recall20 for row in vals if row.test_recall20 is not None]
            ndcg = [row.test_ndcg20 for row in vals if row.test_ndcg20 is not None]
            recall_f = [float(v) for v in recall]
            ndcg_f = [float(v) for v in ndcg]
            r_mean, r_std = mean_std(recall_f)
            n_mean, n_std = mean_std(ndcg_f)
            paired = [
                (row.seed, row.test_recall20, row.test_ndcg20)
                for row in vals
                if row.test_recall20 is not None and row.test_ndcg20 is not None
            ]
            paired_drop = sorted(paired, key=lambda item: item[1])[1:] if len(paired) > 1 else paired
            drop_r = statistics.mean([float(item[1]) for item in paired_drop]) if paired_drop else float("nan")
            drop_n = statistics.mean([float(item[2]) for item in paired_drop]) if paired_drop else float("nan")
            seed_scores = ",".join(f"{seed}:{r:.5f}/{n:.5f}" for seed, r, n in paired)
            f.write(
                f"{candidate}\t{len(vals)}\t{r_mean:.5f}\t{r_std:.5f}\t{n_mean:.5f}\t{n_std:.5f}\t"
                f"{drop_r:.5f}\t{drop_n:.5f}\t{seed_scores}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=f"sports_hparam_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", os.environ.get("GPU", "0")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("SPORTS_EPOCHS", "200")))
    parser.add_argument("--early-stop", type=int, default=int(os.environ.get("SPORTS_EARLY_STOP", "20")))
    parser.add_argument("--strict-probe-test-interval", type=int, default=int(os.environ.get("STRICT_PROBE_TEST_INTERVAL", "10")))
    args = parser.parse_args()

    base_dir = ROOT / "exp_report" / DATASET / "hparam_search" / args.run_tag
    base_dir.mkdir(parents=True, exist_ok=True)

    current = Config()
    selected: dict[str, str] = {}

    search_plan = [
        ("01_batch_size", "batch_size", [128, 256, 512, 1024]),
        ("02_lr_rec", "lr_rec", [0.0005, 0.001, 0.002, 0.005]),
        ("03_modality_bpr_coeff", "modality_bpr_coeff", [0.0, 0.1, 0.2, 0.5, 1.0]),
        ("04_reg_coeff", "reg_coeff", [1e-5, 1e-4, 3e-4, 1e-3]),
        ("05_rec_neighbor_cl_weight", "rec_neighbor_cl_weight", [0.0, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]),
    ]

    for step_dir_name, field_name, values in search_plan:
        step_dir = base_dir / "search" / step_dir_name
        rows: list[RunResult] = []
        for value in values:
            candidate = {
                "batch_size": f"bs{value}",
                "lr_rec": f"lr{slug_float(value)}",
                "modality_bpr_coeff": f"mbpr{slug_float(value)}",
                "reg_coeff": f"reg{slug_float(value)}",
                "rec_neighbor_cl_weight": f"clw{slug_float(value)}",
            }[field_name]
            cfg = current.with_update(**{field_name: value})
            rows.append(
                run_one(
                    phase_dir=step_dir,
                    candidate=candidate,
                    seed=SEARCH_SEED,
                    config=cfg,
                    run_tag=args.run_tag,
                    device_id=args.device_id,
                    dry_run=args.dry_run,
                    epochs=args.epochs,
                    early_stop=args.early_stop,
                    strict_probe_test_interval=args.strict_probe_test_interval,
                )
            )

        write_rows(step_dir / "summary.tsv", rows)
        if args.dry_run:
            continue

        best_candidate = choose_best(
            step_dir_name,
            rows,
            prefer_larger_batch_on_tie=(field_name == "batch_size"),
        )
        selected[field_name] = best_candidate
        best_row = next(row for row in rows if row.candidate == best_candidate)
        value_by_candidate = {
            {
                "batch_size": f"bs{value}",
                "lr_rec": f"lr{slug_float(value)}",
                "modality_bpr_coeff": f"mbpr{slug_float(value)}",
                "reg_coeff": f"reg{slug_float(value)}",
                "rec_neighbor_cl_weight": f"clw{slug_float(value)}",
            }[field_name]: value
            for value in values
        }
        current = current.with_update(**{field_name: value_by_candidate[best_candidate]})
        log(f"selected {field_name}={value_by_candidate[best_candidate]} val_R20={best_row.val_recall20:.5f}")

        if field_name == "rec_neighbor_cl_weight":
            no_cl = next((row for row in rows if row.candidate == "clw0"), None)
            if no_cl and no_cl.val_recall20 is not None and best_row.val_recall20 is not None:
                gain = best_row.val_recall20 - no_cl.val_recall20
                (step_dir / "cl_gain_vs_no_cl.txt").write_text(f"{gain:.6f}\n")
                if current.rec_neighbor_cl_weight > 0 and gain < TIE_EPS:
                    log(f"CL gain {gain:.6f} < {TIE_EPS}; selecting no-CL for final best config")
                    current = current.with_update(rec_neighbor_cl_weight=0.0)

    (base_dir / "selected_config.txt").write_text(str(current) + "\n")
    if args.dry_run or args.skip_final:
        log(f"done search setup run_tag={args.run_tag}")
        return 0

    final_dir = base_dir / "final"
    no_cl = current.with_update(rec_neighbor_cl_weight=0.0)
    mainline = Config()
    final_configs = {
        "best": current,
        "no_cl": no_cl,
        "mainline": mainline,
    }
    rows = []
    seen: set[tuple] = set()
    for candidate, cfg in final_configs.items():
        key = dataclasses.astuple(cfg)
        if key in seen:
            log(f"skip duplicate final config candidate={candidate}")
            continue
        seen.add(key)
        for seed in FULL_SEEDS:
            rows.append(
                run_one(
                    phase_dir=final_dir,
                    candidate=candidate,
                    seed=seed,
                    config=cfg,
                    run_tag=args.run_tag,
                    device_id=args.device_id,
                    dry_run=False,
                    epochs=args.epochs,
                    early_stop=args.early_stop,
                    strict_probe_test_interval=args.strict_probe_test_interval,
                )
            )

    write_rows(final_dir / "summary.tsv", rows)
    write_final_summary(final_dir / "aggregate.tsv", rows)
    log(f"all done; summary={final_dir / 'aggregate.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
