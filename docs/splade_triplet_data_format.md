<!---
Copyright 2025 BizReach, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# SPLADE Triplet-based Input Data Specification

This document describes the triplet-based input data format required for training SPLADE v2 (Sparse Lexical and Expansion) models using the light-splade framework.

## Overview

SPLADE triplet-based dataset consists of four types of data files, all in **NDJSON (Newline Delimited JSON)** format:

1. **Query Master** - Contains query texts and their IDs
2. **Document Master** - Contains document texts and their IDs
3. **Positive Lists** - Maps queries to their relevant documents
4. **Triplets** - Contains training triplets (query, positive doc, negative doc)

## File Format: NDJSON

All data files must be in NDJSON format, where:
- Each line is a valid JSON object
- Lines are separated by newline characters (`\n`)
- Each file can contain multiple JSON objects, one per line
- File extension is `.ndjson` for raw text.
- File can be gzip-compressed for space efficiency (`.ndjson.gz`)

## Data File Specifications

### 1. Query Master File

**File naming convention**: `query_master.ndjson`

**Purpose**: Stores all query texts with their unique identifiers.

**Schema**
```json
{
  "qid": <integer>,
  "text": <string>
}
```

**Fields**

- `qid` (int): Unique query identifier
- `text` (str): The query text

**Example**

```json
{"qid": 3, "text": "Gitでコミットを取り消すにはどうすればよい？"}
{"qid": 4, "text": "睡眠の質を改善するための習慣を教えて。"}
{"qid": 5, "text": "Dockerコンテナのリソース使用量を制限したい。"}
```

### 2. Document Master File

**File naming convention**: `doc_master.ndjson`

**Purpose**: Stores all document texts with their unique identifiers.

**Schema**

```json
{
  "doc_id": <integer>,
  "text": <string>
}
```

**Fields**

- `doc_id` (int): Unique document identifier
- `text` (str): The document text

**Example**

```json
{"doc_id": 11, "text": "直前のコミットを取り消すにはgit reset --softやgit revertを状況に応じて使い分けます。"}
{"doc_id": 12, "text": "雨天時は運動会が体育館で実施される予定です。"}
{"doc_id": 13, "text": "紅葉の名所は朝の光でより美しく見えます。"}
```

### 3. Positive Lists File

**File naming convention**: `positive_lists.ndjson`

**Purpose**: Maps each query to its list of relevant (positive) documents.

**Schema**

```json
{
  "qid": <integer>,
  "positive_doc_ids": [<integer>, ...]
}
```

**Fields**

- `qid` (int): Query identifier (must exist in query master)
- `positive_doc_ids` (list[int]): List of document IDs that are relevant to this query (must exist in document master)

**Example**:
```json
{"qid": 3, "positive_doc_ids": [11, 100]}
{"qid": 4, "positive_doc_ids": [16]}
{"qid": 5, "positive_doc_ids": [21]}
```

**Requirements**

- Each query must have **at least one** positive document
- All document IDs must exist in the document master

### 4. Triplets File

**File naming convention**: `triplets.ndjson`

**Purpose**: Contains training triplets for contrastive learning. Each triplet consists of a query, a positive (relevant) document, and a negative (non-relevant) document.

**Schema**

```json
{
  "qid": <integer>,
  "pos_doc_id": <integer>,
  "neg_doc_id": <integer>
}
```

**Fields**

- `qid` (int): Query identifier (must exist in query master)
- `pos_doc_id` (int): Positive document ID (must exist in document master and in the positive list for this query)
- `neg_doc_id` (int): Negative document ID (must exist in document master)

**Example**

```json
{"qid": 3, "pos_doc_id": 11, "neg_doc_id": 12}
{"qid": 4, "pos_doc_id": 16, "neg_doc_id": 17}
{"qid": 4, "pos_doc_id": 16, "neg_doc_id": 18}
{"qid": 5, "pos_doc_id": 21, "neg_doc_id": 22}
```

**Notes**

- The number of triplets determines the dataset size for training
- Multiple triplets can share the same query ID
- The positive document must be in the query's positive list
- This triplet data is available in training set only
- As in contrastive learning, the relative distance between a query and its positive document versus negative documents is crucial. The way positive and negative document pairs are selected to form triplets can significantly affect model quality.

## Data Organization

### Directory Structure

For a complete training setup, organize your data as follows:

```
data/
├── train/
│   ├── query_master.ndjson
│   ├── doc_master.ndjson
│   ├── positive_lists.ndjson
│   └── triplets.ndjson
└── validation/
    ├── query_master.ndjson
    ├── doc_master.ndjson
    └── positive_lists.ndjson
```

### Configuration Example

Reference your data files in the YAML configuration:

```yaml
DATA_PATH: data/mmarco_ja_4_splade_triplet

train_doc_master: ${.DATA_PATH}/train/doc_master.ndjson
train_query_master: ${.DATA_PATH}/train/query_master.ndjson
train_positives: ${.DATA_PATH}/train/positive_lists.ndjson
train_triplets: ${.DATA_PATH}/train/triplets.ndjson

validation_doc_master: ${.DATA_PATH}/validation/doc_master.ndjson
validation_query_master: ${.DATA_PATH}/validation/query_master.ndjson
validation_positives: ${.DATA_PATH}/validation/positive_lists.ndjson
```

We use Hydra configuration, which supports interpolation in YAML files. For example, ${.DATA_PATH} will be dynamically replaced with the value of the key DATA_PATH.

## Data Validation Rules

The `TripletDataset` class enforces the following validation rules:

1. **Query Coverage**: Every query ID in the positive list must exist in the query master
2. **Query Completeness**: Every query ID in the query master must have entries in the positive list
3. **Document Existence**: All document IDs referenced in the positive lists must exist in the document master
4. **Positive Requirement**: Every query must have at least one positive document

If any validation rule fails, the system will raise a `ValueError` with a specific error message including the problematic ID for easy debugging.

## Dataset Output

When loading a sample from the `TripletDataset`, you receive a 3-tuple of strings:

```python
(query_text, positive_doc_text, negative_doc_text)
```

**Example**:
```python
(
  "Gitでコミットを取り消すにはどうすればよい？",
  "直前のコミットを取り消すにはgit reset --softやgit revertを状況に応じて使い分けます。",
  "雨天時は運動会が体育館で実施される予定です。"
)
```

## Use Case

This data format is designed for **SPLADE v2 training without distillation**, using `in_batch_negatives` or `pairwise_contrastive` for contrastive learning. The model learns to:
- Score queries and documents in a sparse lexical space
- Rank positive documents higher than negative documents
- Learn sparse, interpretable representations

## Reference

- Implementation: [triplet_dataset.py](../src/light_splade/data/triplet_dataset.py)
- Data schemas: [schemas/config/data_training.py](../src/light_splade/schemas/config/data_training.py) and [schemas/data/__init__.py](../src/light_splade/schemas/data/__init__.py)
- Example config: [splade_data_mmarco_ja_triplet.yaml](../config/data/toy_triplet_data.yaml)
