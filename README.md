# light-splade

`light-splade` provides a minimal yet extensible PyTorch implementation of `SPLADE`, a family of sparse neural retrievers that expand queries and documents into interpretable sparse representations.

Unlike dense retrievers, SPLADE produces `sparse vectors in the vocabulary space`, making it both `efficient to index` with standard IR engines (e.g., Lucene, Elasticsearch) and `interpretable`, while achieving strong retrieval effectiveness. It was first introduced in the paper “[SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)”.

This repository is designed for

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

    For full run instructions using `uv` and `Docker` commands, see [Getting started](docs/getting_started.md).

- **Convert text to sparse vector with SPLADE model using this package**

```python
import torch
from light_splade.models.splade import SpladeEncoder

# Initialize the encoder
encoder = SpladeEncoder(model_path="bizreach-inc/light-splade-japanese-28M")

# Tokenize input text
corpus = [
    "日本の首都は東京です。",
    "大阪万博は2025年に開催されます。"
]
token_outputs = encoder.tokenizer(corpus, padding=True, return_tensors="pt")

# Generate sparse representation
with torch.inference_mode():
    sparse_vecs = encoder.get_sparse(
        input_ids=token_outputs["input_ids"],
        attention_mask=token_outputs["attention_mask"]
    )

print(sparse_vecs[0])
print(sparse_vecs[1])
```

- **Convert text to sparse vector with SPLADE model using `transformers` package**

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def dense_to_sparse(dense: torch.tensor, idx2token: dict[int, str]) -> list[dict[str, float]]:
    rows, cols = dense.nonzero(as_tuple=True)
    rows = rows.tolist()
    cols = cols.tolist()
    weights = dense[rows, cols].tolist()

    sparse_vecs = [{} for _ in range(dense.size(0))]
    for row, col, weight in zip(rows, cols, weights):
        sparse_vecs[row][idx2token[col]] = round(weight, 2)

    for i in range(len(sparse_vecs)):
        sparse_vecs[i] = dict(sorted(sparse_vecs[i].items(), key=lambda x: x[1], reverse=True))
    return sparse_vecs


MODEL_PATH = "bizreach-inc/light-splade-japanese-28M"
device = "cuda" if torch.cuda.is_available() else "cpu"
transformer = AutoModelForMaskedLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}

corpus = [
    "日本の首都は東京です。",
    "大阪万博は2025年に開催されます。"
]
token_outputs = tokenizer(corpus, padding=True, return_tensors="pt")
attention_mask = token_outputs["attention_mask"].to(device)
token_outputs = {key: value.to(device) for key, value in token_outputs.items()}

with torch.inference_mode():
    outputs = transformer(**token_outputs)
    dense, _ = torch.max(
        torch.log(1 + torch.relu(outputs.logits)) * attention_mask.unsqueeze(-1),
        dim=1,
    )
sparse_vecs = dense_to_sparse(dense, idx2token)

print(sparse_vecs[0])
print(sparse_vecs[1])
```

- **Output**

```
{'首都': 1.83, '日本': 1.82, '東京': 1.78, '中立': 0.73, '都会': 0.69, '駒': 0.68, '州都': 0.67, '首相': 0.64, '足立': 0.62, 'です': 0.61, '都市': 0.54, 'ユニ': 0.54, '京都': 0.52, '国': 0.51, '発表': 0.49, '成田': 0.48, '太陽': 0.45, '藤原': 0.45, '私立': 0.42, '王国': 0.4...}
{'202': 1.61, '開催': 1.49, '大阪': 1.34, '万博': 1.19, '東京': 1.15, '年': 1.1, 'いつ': 1.05, '##5': 1.03, '203': 0.86, '月': 0.8, '期間': 0.79, '高槻': 0.79, '京都': 0.7, '神戸': 0.62, '2024': 0.54, '夢': 0.52, '206': 0.52, '姫路': 0.51, '行わ': 0.49, 'こう': 0.49, '芸術': 0.48...}
```

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
