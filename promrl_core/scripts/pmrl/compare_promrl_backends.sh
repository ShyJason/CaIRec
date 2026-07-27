#!/usr/bin/env bash
set -euo pipefail

# Example:
#   ACTION=train bash scripts/pmrl/compare_promrl_backends.sh
#   ACTION=test bash scripts/pmrl/compare_promrl_backends.sh

ACTION="${ACTION:-train}"
MODELS=(${MODELS:-promrl promrl_mvae promrl_mopoe promrl_smil promrl_kb})
CUDA_SET="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
AUTO_RESUME="${AUTO_RESUME:-true}"

TRAIN_CONFIG="${TRAIN_CONFIG:-./config/pmrl/finetune_cfg/retrieval-audiocaps_msrvtt.json}"
TEST_CONFIG="${TEST_CONFIG:-./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json}"
PRETRAIN_DIR="${PRETRAIN_DIR:-./pretrain_vast}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TEST_LR="${TEST_LR:-5e-5}"
TRAIN_PORT_BASE="${TRAIN_PORT_BASE:-9850}"
TEST_PORT_BASE="${TEST_PORT_BASE:-9860}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BEST_CKPT_NAME="${BEST_CKPT_NAME:-best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-completion_compare}"
TRAIN_FIRST_EVAL="${TRAIN_FIRST_EVAL:-true}"
TEST_FIRST_EVAL="${TEST_FIRST_EVAL:-false}"
SAVE_BEST="${SAVE_BEST:-true}"
TRAIN_TAU1="${TRAIN_TAU1:-0.05}"
TRAIN_TAU2="${TRAIN_TAU2:-0.1}"
TEST_TAU1="${TEST_TAU1:-0.01}"
TEST_TAU2="${TEST_TAU2:-0.1}"
TRAIN_VALID_FREQ="${TRAIN_VALID_FREQ:-8}"
TRAIN_LAMBDA_ITM="${TRAIN_LAMBDA_ITM:-0.1}"
TEST_LAMBDA_ITM="${TEST_LAMBDA_ITM:-0.1}"
FEATURE_INFERENCE="${FEATURE_INFERENCE:-false}"

for idx in "${!MODELS[@]}"; do
    model="${MODELS[$idx]}"
    output_dir="outputs/${OUTPUT_PREFIX}_${model}"
    resume_args=()

    if [[ "${ACTION}" == "train" && "${AUTO_RESUME}" == "true" ]]; then
        ckpt_dir="${output_dir}/ckpt"
        if [[ -d "${ckpt_dir}" ]] && compgen -G "${ckpt_dir}/optimizer_step_*.pt" > /dev/null; then
            resume_args+=(--resume)
            echo "[compare] Resuming ${model} from ${ckpt_dir}"
        else
            echo "[compare] Starting ${model} from base checkpoint ${BASE_CHECKPOINT}"
        fi
    fi

    if [[ "${ACTION}" == "train" ]]; then
        port=$((TRAIN_PORT_BASE + idx))
        CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
            --nnodes 1 \
            --node_rank 0 \
            --nproc_per_node "${NPROC_PER_NODE}" \
            --master_port "${port}" \
            ./run.py \
            --learning_rate "${TRAIN_LR}" \
            --checkpointing true \
            --first_eval "${TRAIN_FIRST_EVAL}" \
            --save_best "${SAVE_BEST}" \
            --config "${TRAIN_CONFIG}" \
            --pretrain_dir "${PRETRAIN_DIR}" \
            --checkpoint "${BASE_CHECKPOINT}" \
            "${resume_args[@]}" \
            --model_type "${model}" \
            --tau1 "${TRAIN_TAU1}" \
            --tau2 "${TRAIN_TAU2}" \
            --valid_freq "${TRAIN_VALID_FREQ}" \
            --lambda_itm "${TRAIN_LAMBDA_ITM}" \
            --output_dir "${output_dir}" \
            --log_name "${OUTPUT_PREFIX}_${model}" \
            --feature_inference "${FEATURE_INFERENCE}"
    elif [[ "${ACTION}" == "test" ]]; then
        port=$((TEST_PORT_BASE + idx))
        ckpt="${output_dir}/ckpt/${BEST_CKPT_NAME}"
        CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
            --nnodes 1 \
            --node_rank 0 \
            --nproc_per_node "${NPROC_PER_NODE}" \
            --master_port "${port}" \
            ./run.py \
            --learning_rate "${TEST_LR}" \
            --checkpointing true \
            --first_eval "${TEST_FIRST_EVAL}" \
            --config "${TEST_CONFIG}" \
            --checkpoint "${ckpt}" \
            --pretrain_dir "${PRETRAIN_DIR}" \
            --mode testing \
            --model_type "${model}" \
            --save_best "${SAVE_BEST}" \
            --tau1 "${TEST_TAU1}" \
            --tau2 "${TEST_TAU2}" \
            --lambda_itm "${TEST_LAMBDA_ITM}" \
            --output_dir "outputs/retrieval_${OUTPUT_PREFIX}_${model}" \
            --log_name "test_${OUTPUT_PREFIX}_${model}" \
            --feature_inference "${FEATURE_INFERENCE}"

        python extract_results.py \
            --log-file "outputs/retrieval_${OUTPUT_PREFIX}_${model}/log/log.txt" \
            --prefix "${OUTPUT_PREFIX}_${model}"
    else
        echo "Unsupported ACTION=${ACTION}. Use train or test."
        exit 1
    fi
done
