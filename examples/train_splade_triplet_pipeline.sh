#!/bin/bash

set -e

COMMAND=$1

if [ -z $COMMAND ]; then
    COMMAND="all"
fi

if [ -z $SPLADE_CONFIG_NAME ]; then
    SPLADE_CONFIG_NAME="splade_mmarco_ja_triplet"
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
if [ $COMMAND == "convert-mmarco-ja-triplet" ] || [ $COMMAND == "all" ]; then
    print_title "----- Convert mMARCO-ja dataset into the light-splade triplet-based format...."
    uv run examples/run_convert_mmarco_ja_triplet.py > logs/1_run_convert_mmarco_ja_triplet.txt 2>&1
fi

############################
# Train SPLADE
############################
if [ $COMMAND == "train-splade-triplet" ] || [ $COMMAND == "all" ]; then
    print_title "----- Train SPLADE model directly from triplet dataset..."
    uv run examples/run_train_splade_triplet.py --config-name ${SPLADE_CONFIG_NAME} > logs/2_run_train_splade_triplet.txt 2>&1
fi
