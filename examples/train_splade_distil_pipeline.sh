#!/bin/bash

set -e

COMMAND=$1

if [ -z $COMMAND ]; then
    COMMAND="all"
fi

if [ -z $SPLADE_CONFIG_NAME ]; then
    SPLADE_CONFIG_NAME="splade_mmarco_ja_distil"
fi

echo "COMMAND: ${COMMAND}"
echo "SPLADE_CONFIG_NAME: ${SPLADE_CONFIG_NAME}"

print_title () {
    echo "############################"
    echo "$1"
    echo "############################"
}


############################
# Convert data
############################
if [ $COMMAND == "convert-mmarco-ja-distil" ] || [ $COMMAND == "all" ]; then
    print_title "----- Convert mMARCO-ja data for SPLADE training..."
    uv run examples/run_convert_mmarco_ja_distil.py > logs/1_run_convert_mmarco_ja_distil.txt 2>&1
fi

############################
# Train Cross-Encoder as a teacher model
############################
if [ $COMMAND == "train-cross-encoder" ] || [ $COMMAND == "all" ]; then
    print_title "----- Train Cross-Encoder for SPLADE distillation..."
    uv run examples/run_train_cross_encoder.py --config-file config/cross_encoder_train.yaml > logs/2_run_train_cross_encoder.txt 2>&1
fi

############################
# Predict hard-negative scores for SPLADE distillation
############################
if [ $COMMAND == "predict-cross-encoder" ] || [ $COMMAND == "all" ]; then
    print_title "----- Run Cross-Encoder inference to predict similarity scores for (query/doc) pairs..."
    uv run examples/run_predict_cross_encoder.py --config-file config/cross_encoder_predict.yaml > logs/3_run_predict_cross_encoder.txt 2>&1
fi

############################
# Train SPLADE
############################
if [ $COMMAND == "train-splade-distil" ] || [ $COMMAND == "all" ]; then
    print_title "----- Train SPLADE model using distillation method..."
    uv run examples/run_train_splade_distil.py --config-name ${SPLADE_CONFIG_NAME} > logs/4_run_train_splade_distil.txt 2>&1
fi
