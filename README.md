# light-splade

`light-splade` provides a minimal yet extensible PyTorch implementation of `SPLADE`, a family of sparse neural retrievers that expand queries and documents into interpretable sparse representations.

Unlike dense retrievers, SPLADE produces `sparse vectors in the vocabulary space`, making it both `efficient to index` with standard IR engines (e.g., Lucene, Elasticsearch) and `interpretable`, while achieving strong retrieval effectiveness. It was first introduced in the paper “[SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)”.

This repository is designed for:
- Practitioners wanting to `train SPLADE models on custom corpora`.
- Developers experimenting with `sparse lexical expansion` at scale.
- Researchers looking for a `reference implementation`.

We currently support `SPLADE v2` and `SPLADE++`

## Features
- Training pipeline for SPLADE using PyTorch + HuggingFace Transformers.
- Support for `distillation training` from dense retrievers (e.g., ColBERT, dense BERT).
- Export trained models into sparse representations compatible with IR systems.
- Simple, lightweight, and easy to extend for experiments.

## Setup

- Python 3.11+.
- Recommended: use the `uv` tool to manage the virtual environment (see [Getting started](docs/getting_started.md) document).

Quick setup (recommended):

```bash
git clone https://github.com/bizreach-inc/light-splade.git
cd light-splade
# create and activate virtual env using uv
uv venv --seed .venv
source .venv/bin/activate
uv sync
```

For developer checks, run:

```bash
uv run pre-commit run --all-files
uv run pytest
```

## Quickstart

- **Train SPLADE with toy dataset (triplet-based)**:
  - `uv run examples/run_train_splade_triplet.py --config-name toy_splade_ja`
  - To run on an environment without GPU, see this [trouble shooting](docs/trouble_shooting.md#running-the-training-script-on-cpu-only-machines)

- **Convert text to sparse vector using SPLADE model**
  ```python
  import torch
  from light_splade.models.splade import SpladeEncoder

  encoder = SpladeEncoder(model_path="path/to/model")

  token_outputs = encoder.tokenizer("日本の首都は東京です。", return_tensors="pt")
  with torch.inference_mode():
      sparse_vecs = encoder.get_sparse(
          input_ids=token_outputs["input_ids"],
          attention_mask=token_outputs["attention_mask"]
      )

  print(sparse_vecs[0])
  ```

  Output:
  ```
  > {'東京': 2.07, '首都': 1.89, '日本': 1.81, '都': 1.51, '江戸': 1.51, '都市': 1.42, '駐日': 1.37, 'ニッポン': 1.2, '北京': 0.93, '所在': 0.91, '大阪': 0.78, 'ロンドン': 0.68, '天皇': 0.6, 'ソウル': 0.54, '京都': 0.53, '明治': 0.52, '札幌': 0.5, '省': 0.48, 'ジャ': 0.45, 'シティ': 0.42, 'ステート': 0.42, '市': 0.41, '国': 0.39,...}

  (43 tokens)
  ```

For full run instructions using `uv` and `Docker` commands, see [Getting started](docs/getting_started.md).

## Input Data format

Detailed data format docs:

- [Triplet format](docs/splade_triplet_data_format.md) (`SPLADE v2`)
- [Distillation format](docs/splade_triplet_distil_data_format.md) (`SPLADE++` or `SPLADE v2bis`)

## References

- [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086). arxiv (SPLADE v2)
  - Thibault Formal, Benjamin Piwowarski, Carlos Lassance, Stéphane Clinchant.

- [From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective](http://arxiv.org/abs/2205.04733). SIGIR22 short paper (SPLADE++ or SPLADE v2bis)
  - Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant.

- For `transformers` docs:
  - [Trainer docs (transformers v4.56.1)](https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer)
  - [TrainingArguments docs (transformers v4.56.1)](https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer#transformers.TrainingArguments)


## License

This project is licensed under the Apache License, Version 2.0 — see the `LICENSE` file for details.

Copyright 2025 BizReach, Inc.
