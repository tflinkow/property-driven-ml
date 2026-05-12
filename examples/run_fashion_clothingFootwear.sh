#!/usr/bin/env/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 SEED"
    exit 1
fi

SEED="$1"

## CONFIG ##
DATASET="fashion"
CONSTRAINT="ClothingFootwear"
EXPERIMENT_NAME="clothing-footwear"

EPOCHS=100
BATCH_SIZE=512
LR=1e-3

PGD_ITERATIONS=20
PGD_RESTARTS=2

RESULT_DIR="results/${EXPERIMENT_NAME}/${DATASET}/${SEED}"
SPEC="fashion.vcl"

# TODO: specify path to Marabou here
MARABOU="/home/thomasflinkow/Marabou/build/Marabou"
TIMEOUT=30

LOGICS=("QLL_5" "STL_5" "DL2" "Baseline" "QLL_1" "QLL_2" "QLL_10" "STL_1" "STL_2" "STL_10")
EPS_VALUES=("0.05" "0.1")

ALPHA=0.5

mkdir -p "${RESULT_DIR}"

## HELPERS ##
eps_name() {
    local val="$1"
    echo "${val//./_}"
}

train_model() {
    local logic="$1"

    if [[ "${logic}" == "Baseline" ]]; then
        python ../main.py \
            --dataset="${DATASET}" \
            --epochs="${EPOCHS}" \
            --batch-size="${BATCH_SIZE}" \
            --lr="${LR}" \
            --save-onnx \
            --save-imgs \
            --epsilon=0.1 \
            --oracle-steps="${PGD_ITERATIONS}" \
            --oracle-restarts="${PGD_RESTARTS}" \
            --constraint="${CONSTRAINT}" \
            --alpha="${ALPHA}" \
            --seed="${SEED}"
    else
        python ../main.py \
            --dataset="${DATASET}" \
            --epochs="${EPOCHS}" \
            --batch-size="${BATCH_SIZE}" \
            --lr="${LR}" \
            --save-onnx \
            --save-imgs \
            --epsilon=0.1 \
            --oracle-steps="${PGD_ITERATIONS}" \
            --oracle-restarts="${PGD_RESTARTS}" \
            --constraint="${CONSTRAINT}" \
            --seed="${SEED}" \
            --logic="${logic}"
    fi
}

verify() {
    local logic="$1"
    local eps="$2"

    local eps_name
    eps_name="$(eps_name "${eps}")"

    local onnx="${RESULT_DIR}/${logic}.onnx"
    local images="fashion-images-100.idx"
    local labels="fashion-labels-100.idx"
    local log="${RESULT_DIR}/${logic}_${eps_name}.log"

    vehicle verify \
        --no-sat-print \
        -s "${SPEC}" \
        -y clothingFootwear \
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
# python mnist_to_idx.py

## TRAIN ##
for logic in "${LOGICS[@]}"; do
    train_model "${logic}"
done

## VERIFY ##
for logic in "${LOGICS[@]}"; do
    for eps in "${EPS_VALUES[@]}"; do
        verify "${logic}" "${eps}"
    done
done