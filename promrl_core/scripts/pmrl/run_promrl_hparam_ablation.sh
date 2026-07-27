#!/usr/bin/env bash
set -euo pipefail

CUDA_SET="${CUDA_VISIBLE_DEVICES:-5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_PORT_BASE="${TRAIN_PORT_BASE:-9890}"
TEST_PORT_BASE="${TEST_PORT_BASE:-9910}"

TRAIN_CONFIG="${TRAIN_CONFIG:-./config/pmrl/finetune_cfg/retrieval-audiocaps2.json}"
TEST_CONFIG="${TEST_CONFIG:-./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json}"
PRETRAIN_DIR="${PRETRAIN_DIR:-./pretrain_vast}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-./outputs/pretrain_promrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
VALID_FREQ="${VALID_FREQ:-32}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-promrl_hparam_firstval_$(date -u +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-results}"
BEST_CKPT_NAME="${BEST_CKPT_NAME:-best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt}"
FIRST_VAL_TIMEOUT_SEC="${FIRST_VAL_TIMEOUT_SEC:-10800}"
POLL_SEC="${POLL_SEC:-20}"
STOP_AFTER_FIRST_VAL="${STOP_AFTER_FIRST_VAL:-true}"
ABLATION_GROUPS="${ABLATION_GROUPS:-tau1,tau2,lambda_itm}"

mkdir -p "${RESULTS_DIR}"
mkdir -p "outputs/queue_logs"

SUMMARY_TSV="${RESULTS_DIR}/${OUTPUT_PREFIX}_overall_r1.tsv"
SUMMARY_MD="${RESULTS_DIR}/${OUTPUT_PREFIX}_overall_r1.md"

if [[ ! -s "${SUMMARY_TSV}" ]]; then
    printf "group\tvalue\toverall_r1\telapsed_sec\ttrain_dir\ttest_log\n" > "${SUMMARY_TSV}"
fi

write_summary_md() {
    python - <<'PY' "${SUMMARY_TSV}" "${SUMMARY_MD}"
import csv
import sys

tsv_path, md_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(tsv_path, newline="", encoding="utf-8"), delimiter="\t"))
lookup = {(row["group"], row["value"]): row["overall_r1"] for row in rows}

with open(md_path, "w", encoding="utf-8") as f:
    f.write("| $\\\\tau$ | 0.02 | 0.2 | 1 | 2 | $\\\\tau'$ | 0.02 | 0.2 | 1 | 2 |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|\n")
    f.write(
        "| Overall R@1 | "
        + " | ".join(lookup.get(("tau1", v), "XX") for v in ["0.02", "0.2", "1", "2"])
        + " || "
        + " | ".join(lookup.get(("tau2", v), "XX") for v in ["0.02", "0.2", "1", "2"])
        + " |\n\n"
    )
    f.write("| $\\\\alpha$ | 0 | 0.1 | 0.5 | 1 |\n")
    f.write("|---|---|---|---|---|\n")
    f.write(
        "| Overall R@1 | "
        + " | ".join(lookup.get(("lambda_itm", v), "XX") for v in ["0", "0.1", "0.5", "1"])
        + " |\n"
    )
PY
}

checkpoint_stable() {
    local best_ckpt="$1"
    local ckpt_dir="$2"

    [[ -f "${best_ckpt}" ]] || return 1

    local model_ckpt
    model_ckpt="$(ls -1 "${ckpt_dir}"/model_step_*.pt 2>/dev/null | tail -n 1 || true)"
    [[ -n "${model_ckpt}" ]] || return 1

    local best_size_1 model_size_1 best_size_2 model_size_2
    best_size_1="$(stat -c %s "${best_ckpt}")"
    model_size_1="$(stat -c %s "${model_ckpt}")"
    sleep "${POLL_SEC}"
    best_size_2="$(stat -c %s "${best_ckpt}")"
    model_size_2="$(stat -c %s "${model_ckpt}")"

    [[ "${best_size_1}" == "${best_size_2}" ]] || return 1
    [[ "${model_size_1}" == "${model_size_2}" ]] || return 1

    return 0
}

wait_for_first_validation() {
    local pid="$1"
    local best_ckpt="$2"
    local ckpt_dir="$3"
    local started_at="$4"

    while kill -0 "${pid}" 2>/dev/null; do
        if checkpoint_stable "${best_ckpt}" "${ckpt_dir}"; then
            return 0
        fi

        local now
        now="$(date +%s)"
        if (( now - started_at > FIRST_VAL_TIMEOUT_SEC )); then
            echo "[ablation] timeout while waiting for first validation checkpoint: ${best_ckpt}" >&2
            return 1
        fi

        sleep "${POLL_SEC}"
    done

    if checkpoint_stable "${best_ckpt}" "${ckpt_dir}"; then
        return 0
    fi

    return 1
}

run_one() {
    local group="$1"
    local value="$2"
    local tau1="$3"
    local tau2="$4"
    local lambda_itm="$5"
    local idx="$6"
    local label="${group}_${value}"
    local train_dir="outputs/${OUTPUT_PREFIX}_${label}"
    local test_dir="outputs/${OUTPUT_PREFIX}_zeroshot_${label}"
    local train_port=$((TRAIN_PORT_BASE + idx))
    local test_port=$((TEST_PORT_BASE + idx))
    local ckpt_dir="${train_dir}/ckpt"
    local best_ckpt="${train_dir}/ckpt/${BEST_CKPT_NAME}"

    echo "[ablation] ${label} tau1=${tau1} tau2=${tau2} lambda_itm=${lambda_itm}"
    rm -rf "${train_dir}" "${test_dir}"

    local started_at
    started_at="$(date +%s)"

    if [[ "${STOP_AFTER_FIRST_VAL}" == "true" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
            --nnodes 1 \
            --node_rank 0 \
            --nproc_per_node "${NPROC_PER_NODE}" \
            --master_port "${train_port}" \
            ./run.py \
            --learning_rate "${LEARNING_RATE}" \
            --checkpointing true \
            --first_eval false \
            --save_best true \
            --config "${TRAIN_CONFIG}" \
            --pretrain_dir "${PRETRAIN_DIR}" \
            --checkpoint "${BASE_CHECKPOINT}" \
            --model_type promrl \
            --tau1 "${tau1}" \
            --tau2 "${tau2}" \
            --valid_freq "${VALID_FREQ}" \
            --lambda_itm "${lambda_itm}" \
            --output_dir "${train_dir}" \
            --log_name "${OUTPUT_PREFIX}_${label}" \
            --feature_inference false &
        local train_pid=$!

        if ! wait_for_first_validation "${train_pid}" "${best_ckpt}" "${ckpt_dir}" "${started_at}"; then
            echo "[ablation] failed before first validation completed for ${label}" >&2
            kill -TERM "${train_pid}" 2>/dev/null || true
            wait "${train_pid}" || true
            return 1
        fi

        echo "[ablation] first validation finished for ${label}, stopping training early"
        kill -TERM "${train_pid}" 2>/dev/null || true
        wait "${train_pid}" || true
    else
        CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
            --nnodes 1 \
            --node_rank 0 \
            --nproc_per_node "${NPROC_PER_NODE}" \
            --master_port "${train_port}" \
            ./run.py \
            --learning_rate "${LEARNING_RATE}" \
            --checkpointing true \
            --first_eval false \
            --save_best true \
            --config "${TRAIN_CONFIG}" \
            --pretrain_dir "${PRETRAIN_DIR}" \
            --checkpoint "${BASE_CHECKPOINT}" \
            --model_type promrl \
            --tau1 "${tau1}" \
            --tau2 "${tau2}" \
            --valid_freq "${VALID_FREQ}" \
            --lambda_itm "${lambda_itm}" \
            --output_dir "${train_dir}" \
            --log_name "${OUTPUT_PREFIX}_${label}" \
            --feature_inference false
    fi

    if [[ ! -f "${best_ckpt}" ]]; then
        echo "[ablation] missing checkpoint after first validation: ${best_ckpt}" >&2
        return 1
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
        --nnodes 1 \
        --node_rank 0 \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --master_port "${test_port}" \
        ./run.py \
        --learning_rate "${LEARNING_RATE}" \
        --checkpointing true \
        --first_eval false \
        --save_best true \
        --config "${TEST_CONFIG}" \
        --pretrain_dir "${PRETRAIN_DIR}" \
        --checkpoint "${best_ckpt}" \
        --model_type promrl \
        --mode testing \
        --tau1 "${tau1}" \
        --tau2 "${tau2}" \
        --valid_freq "${VALID_FREQ}" \
        --lambda_itm "${lambda_itm}" \
        --output_dir "${test_dir}" \
        --log_name "${OUTPUT_PREFIX}_zeroshot_${label}" \
        --feature_inference false

    local overall_json
    overall_json="$(python scripts/pmrl/extract_overall_r1.py "${test_dir}/log/log.txt")"
    local overall_r1
    overall_r1="$(python - <<'PY' "${overall_json}"
import json
import sys
print(json.loads(sys.argv[1])["overall_r1"])
PY
)"

    local finished_at elapsed_sec
    finished_at="$(date +%s)"
    elapsed_sec="$((finished_at - started_at))"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${group}" "${value}" "${overall_r1}" "${elapsed_sec}" "${train_dir}" "${test_dir}/log/log.txt" \
        >> "${SUMMARY_TSV}"
    write_summary_md
}

idx=0
if [[ ",${ABLATION_GROUPS}," == *",tau1,"* ]]; then
    for value in 0.02 0.2 1 2; do
        run_one "tau1" "${value}" "${value}" "0.1" "0.1" "${idx}"
        idx=$((idx + 1))
    done
fi

if [[ ",${ABLATION_GROUPS}," == *",tau2,"* ]]; then
    for value in 0.02 0.2 1 2; do
        run_one "tau2" "${value}" "0.05" "${value}" "0.1" "${idx}"
        idx=$((idx + 1))
    done
fi

if [[ ",${ABLATION_GROUPS}," == *",lambda_itm,"* ]]; then
    for value in 0 0.5 1; do
        run_one "lambda_itm" "${value}" "0.05" "0.1" "${value}" "${idx}"
        idx=$((idx + 1))
    done
fi

echo "[ablation] summary_tsv=${SUMMARY_TSV}"
echo "[ablation] summary_md=${SUMMARY_MD}"
