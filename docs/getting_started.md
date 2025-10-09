# Getting started

This document contains quick run instructions on how to run the repository using `uv` and using Docker.

## Run with uv

### Install uv

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more information on uv, please refer to <https://github.com/astral-sh/uv>

### Install Virtual Environment with `uv`

```shell
cd light-splade
uv venv --seed .venv
source .venv/bin/activate
uv sync
```

### Run lint via pre-commit under uv

```shell
uv run pre-commit run --all-files
```

Pre-commit is configured in the `light-splade/.pre-commit-config.yaml` file.

### Run Unit Tests under uv

```shell
uv run pytest
```

## Train with mMARCO-ja (triplet-only)

### Full pipeline

```shell
export SPLADE_CONFIG_NAME=splade_mmarco_ja_triplet
nohup examples/train_splade_triplet_pipeline.sh &
```

**Note**: This pipeline includes data building and several model training step, so a GPU is required.

### Step-by-step

**STEP 1** — convert the mMARCO-ja dataset into light-splade triplet format:

```shell
nohup uv run examples/run_convert_mmarco_ja_triplet.py > logs/1_run_convert_mmarco_ja_triplet.txt 2>&1 &
```

**STEP 2** — train SPLADE from triplet dataset:

```shell
nohup uv run examples/run_train_splade_triplet.py --config-name splade_mmarco_ja_triplet > logs/2_run_train_splade_triplet.txt 2>&1 &
```

## Train with mMARCO-ja (distillation)

### Full pipeline

```shell
export SPLADE_CONFIG_NAME=splade_mmarco_ja_distil
nohup examples/train_splade_distil_pipeline.sh &
```

**Note**: This pipeline includes several model training steps, so a GPU is required.

### Step-by-step

**STEP 1** — convert the mMARCO-ja dataset into light-splade distil format:

```shell
nohup uv run examples/run_convert_mmarco_ja_distil.py > logs/1_run_convert_mmarco_ja_distil.txt 2>&1 &
```

**STEP 2** — train a Cross-Encoder (teacher):

```shell
nohup uv run examples/run_train_cross_encoder.py --config-file config/cross_encoder_train.yaml > logs/2_run_train_cross_encoder.txt 2>&1 &
```

**STEP 3** — infer similarity scores with Cross-Encoder:

```shell
nohup uv run examples/run_predict_cross_encoder.py --config-file config/cross_encoder_predict.yaml > logs/3_run_predict_cross_encoder.txt 2>&1 &
```

**STEP 4** — train SPLADE using predicted similarity scores:

```shell
nohup uv run examples/run_train_splade_distil.py --config-name splade_mmarco_ja_distil > logs/4_run_train_splade_distil.txt 2>&1 &
```

## Run with Docker

### Build Docker image

```shell
cd light-splade
docker compose build
```

## Train with Docker — triplet (no distillation)

### Full pipeline

```shell
docker compose up all-triplet -d
```

### Step-by-step

**STEP 1** — convert mMARCO-ja collection:

```shell
docker compose up convert-mmarco-ja-triplet -d
```

**STEP 2** — train SPLADE from triplets:

```shell
docker compose up train-splade-triplet -d
```

## Train with Docker — distillation

### Full pipeline

```shell
docker compose up all-distil -d
```

### Step-by-step

**STEP 1** — convert mMARCO-ja collection

```shell
docker compose up convert-mmarco-ja-distil -d
```

**STEP 2** — train Cross-Encoder (teacher)

```shell
docker compose up train-cross-encoder -d
```

**STEP 3** — predict similarity with Cross-Encoder

```shell
docker compose up predict-cross-encoder -d
```

**STEP 4** — train SPLADE using predicted similarity scores

```shell
docker compose up train-splade-distil -d
```
