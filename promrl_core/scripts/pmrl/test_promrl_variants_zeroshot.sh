#!/usr/bin/env bash
set -euo pipefail

MODELS=(${MODELS:-promrl promrl_mvae promrl_mopoe promrl_smil promrl_kb})
CUDA_SET="${CUDA_VISIBLE_DEVICES:-5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PORT_BASE="${PORT_BASE:-9870}"

CONFIG="${CONFIG:-./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json}"
PRETRAIN_DIR="${PRETRAIN_DIR:-./pretrain_vast}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TAU1="${TAU1:-0.05}"
TAU2="${TAU2:-0.1}"
LAMBDA_ITM="${LAMBDA_ITM:-0}"
VALID_FREQ="${VALID_FREQ:-32}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-retrieval_all_zeroshot_variants}"

declare -A CHECKPOINTS=(
    [promrl]="./outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_audiocaps/ckpt/best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt"
    [promrl_mvae]="./outputs/compare_audiocaps2_20260329_163538_promrl_mvae/ckpt/best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt"
    [promrl_mopoe]="./outputs/compare_audiocaps2_20260329_163538_promrl_mopoe/ckpt/best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt"
    [promrl_smil]="./outputs/compare_audiocaps2_20260329_163538_promrl_smil/ckpt/best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt"
    [promrl_kb]="./outputs/compare_audiocaps2_20260329_163538_promrl_kb/ckpt/best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt"
)

for idx in "${!MODELS[@]}"; do
    model="${MODELS[$idx]}"
    checkpoint="${CHECKPOINTS[$model]:-}"
    output_dir="outputs/${OUTPUT_PREFIX}_${model}"
    port=$((PORT_BASE + idx))

    if [[ -z "${checkpoint}" ]]; then
        echo "No checkpoint configured for model=${model}" >&2
        exit 1
    fi

    if [[ ! -f "${checkpoint}" ]]; then
        echo "Checkpoint not found for model=${model}: ${checkpoint}" >&2
        exit 1
    fi

    echo "[zeroshot] model=${model}"
    echo "[zeroshot] checkpoint=${checkpoint}"
    echo "[zeroshot] output_dir=${output_dir}"

    CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
        --nnodes 1 \
        --node_rank 0 \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --master_port "${port}" \
        ./run.py \
        --learning_rate "${LEARNING_RATE}" \
        --checkpointing true \
        --first_eval false \
        --save_best true \
        --config "${CONFIG}" \
        --pretrain_dir "${PRETRAIN_DIR}" \
        --output_dir "${output_dir}" \
        --checkpoint "${checkpoint}" \
        --model_type "${model}" \
        --mode testing \
        --tau1 "${TAU1}" \
        --tau2 "${TAU2}" \
        --valid_freq "${VALID_FREQ}" \
        --lambda_itm "${LAMBDA_ITM}" \
        --log_name "${OUTPUT_PREFIX}_${model}" \
        --feature_inference false

    python extract_results.py \
        --log-file "${output_dir}/log/log.txt" \
        --prefix "${OUTPUT_PREFIX}_${model}"
done
