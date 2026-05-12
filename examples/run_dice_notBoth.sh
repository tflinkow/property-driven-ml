#!/usr/bin/env/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 SEED"
    exit 1
fi

SEED="$1"

## CONFIG ##
DATASET="dice"
CONSTRAINT="NotBoth"
EXPERIMENT_NAME="not-both"

EPOCHS=100
BATCH_SIZE=16
LR=1e-3

PGD_ITERATIONS=30
PGD_RESTARTS=3

RESULT_DIR="results/${EXPERIMENT_NAME}/${DATASET}/${SEED}"
SPEC="dice.vcl"

# TODO: specify path to Marabou here
MARABOU="/home/thomasflinkow/Marabou/build/Marabou"
TIMEOUT=30

CHUNK_SIZE=68
CHUNKS=(0)

LOGICS=("QLL_5" "STL_5" "DL2" "Baseline" "RC" "YG" "GD" "LK" "YG" "QLL_1" "QLL_2" "QLL_10" "STL_1" "STL_2" "STL_10")
EPS_NUMS=("2" "4")

ALPHA=0.5

mkdir -p "${RESULT_DIR}"

## HELPERS ##
eps_value() {
    local num="$1"
    python -c "num=${num}; print(f'{num/255:.10f}')"
}

train_model() {
    local logic="$1"

    if [[ "${logic}" == "Baseline" ]]; then
        # eps =  4/255 = 0.0156862745
        python ../main.py \
            --dataset="${DATASET}" \
            --epochs="${EPOCHS}" \
            --batch-size="${BATCH_SIZE}" \
            --lr="${LR}" \
            --save-onnx \
            --save-imgs \
            --epsilon=0.0156862745 \
            --oracle-steps="${PGD_ITERATIONS}" \
            --oracle-restarts="${PGD_RESTARTS}" \
            --constraint="${CONSTRAINT}" \
            --alpha="${ALPHA}" \
            --seed="${SEED}"
    else
        # eps =  4/255 = 0.0156862745
        python ../main.py \
            --dataset="${DATASET}" \
            --epochs="${EPOCHS}" \
            --batch-size="${BATCH_SIZE}" \
            --lr="${LR}" \
            --save-onnx \
            --save-imgs \
            --epsilon=0.0156862745 \
            --oracle-steps="${PGD_ITERATIONS}" \
            --oracle-restarts="${PGD_RESTARTS}" \
            --constraint="${CONSTRAINT}" \
            --alpha="${ALPHA}" \
            --seed="${SEED}" \
            --logic="${logic}"
    fi
}

verify_chunk() {
    local logic="$1"
    local eps_num="$2"
    local chunk="$3"

    local eps_name="${eps_num}_255"
    local eps
    eps="$(eps_value "${eps_num}")"

    local onnx="${RESULT_DIR}/${logic}.onnx"
    local images="dice-images-size${CHUNK_SIZE}-chunk${chunk}.idx"
    local labels="dice-labels-size${CHUNK_SIZE}-chunk${chunk}.idx"
    local log="${RESULT_DIR}/${logic}_${eps_name}_chunk${chunk}.log"

    vehicle verify \
        --no-sat-print \
        -s "${SPEC}" \
        -y noOppositePair \
        -n "classifier:${onnx}" \
        -p "epsilon:${eps}" \
        -d "images:${images}" \
        -d "labels:${labels}" \
        -v Marabou \
        -a "--timeout=${TIMEOUT}" \
        -l "${MARABOU}" \
        2>&1 | tee "${log}"
}

## CREATE IDX FILES ##
#PYTHONPATH=.. python dice_to_idx.py

## TRAIN ##
for logic in "${LOGICS[@]}"; do
   train_model "${logic}"
done

## VERIFY ##
for logic in "${LOGICS[@]}"; do
    for eps_num in "${EPS_NUMS[@]}"; do
        for chunk in "${CHUNKS[@]}"; do
            verify_chunk "${logic}" "${eps_num}" "${chunk}"
        done
    done
done
