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

# SPLADE Distillation-based Input Data Specification

This document describes the input data format required for training SPLADE++ (Sparse Lexical and Expansion) models **with knowledge distillation** from a teacher model using the light-splade framework.

## Overview

SPLADE distillation-based dataset consists of four types of data files, all in **NDJSON (Newline Delimited JSON)** format:

1. **Query Master** - Contains query texts and their IDs
2. **Document Master** - Contains document texts and their IDs
3. **Positive Lists** - Maps queries to their relevant documents
4. **Hard Negative Scores** - Contains similarity scores from a teacher model (e.g., cross-encoder) for query-document pairs

## Key Differences from Standard Triplet Training

Unlike standard triplet training, the distillation approach:
- Does **NOT** require pre-generated triplets file
- Samples positive and negative documents **dynamically** during training
- Requires **similarity scores from a teacher model** for distillation
- Augments training with soft labels from the teacher model
- Uses hard negatives (documents with high teacher scores but not marked as positive)

## File Format: NDJSON

All data files must be in NDJSON format, where:
- Each line is a valid JSON object
- Lines are separated by newline characters (`\n`)
- Each file can contain multiple JSON objects, one per line
- Files can be optionally gzip-compressed (`.ndjson.gz`)

## Data File Specifications

### 1. Query Master File

Same as the Query Master File in [Triplet-based data format](splade_triplet_data_format.md###1-query-master-file)


### 2. Document Master File

Same as the Document Master File in [Triplet-based data format](splade_triplet_data_format.md###2-document-master-file)

### 3. Positive Lists File

Same as the Positive Lists File in [Triplet-based data format](splade_triplet_data_format.md###3-positive-lists-file)

### 4. Hard Negative Scores File

**File naming convention**: `hard-negatives-cross-encoder-scores.ndjson` or `hard_negative_scores.ndjson`

**Purpose**: Contains similarity scores from a teacher model (typically a strong cross-encoder) for query-document pairs. These scores are used for:
1. **Knowledge distillation** - The student model learns to mimic the teacher's scoring behavior
2. **Hard negative mining** - Selecting challenging negative examples (high teacher score but not marked as positive)

**Schema**

```json
{
  "qid": <integer>,
  "scores": {
    "<doc_id>": <float>,
    "<doc_id>": <float>,
    ...
  }
}
```

**Fields**

- `qid` (int): Query identifier (must exist in query master)
- `scores` (dict[int, float]): Dictionary mapping document IDs to their similarity scores from the teacher model
    - Keys can be stored as strings or integers in JSON (will be converted to integers internally)
    - Values are floating-point scores from the teacher model

**Example**:
```json
{"qid": 3, "scores": {"11": 0.95, "12": 0.23, "13": 0.15, "100": 0.26}}
{"qid": 4, "scores": {"16": 0.92, "17": 0.31, "18": 0.42}}
{"qid": 5, "scores": {"21": 0.89, "22": 0.18, "23": 0.27}}
```

**Requirements**

- Every query in the positive list must have entries in the scores file
- For each query, **all positive documents** must have scores
- Scores should include hard negative candidates (documents with relatively high scores but not in the positive list)
- The more candidate documents with scores, the better the hard negative mining

**Notes on Teacher Scores**

- Teacher model is typically a cross-encoder (e.g., cross-encoder/ms-marco-MiniLM-L-12-v2)
- Scores represent the teacher's assessment of query-document relevance
- Higher scores indicate higher relevance according to the teacher
- Hard negatives are selected from documents with high teacher scores that are NOT in the positive list

## Data Organization

### Directory Structure

For a complete training setup with distillation, organize your data as follows:

```
data/
├── train/
│   ├── query_master.ndjson
│   ├── doc_master.ndjson
│   └── positive_lists.ndjson
├── validation/
│   ├── query_master.ndjson
│   ├── doc_master.ndjson
│   └── positive_lists.ndjson
└── hard-negatives-cross-encoder-scores.ndjson.gz
```

**Important Notes**

- Hard negative scores file is typically **shared** between train and validation sets (placed at the root)
- No separate triplets file is needed (triplets are sampled dynamically)
- The scores file can be large, so gzip compression is recommended

### Configuration Example

Reference your data files in the YAML configuration:

```yaml
DATA_PATH: data/mmarco_ja_4_splade_distil

train_doc_master: ${.DATA_PATH}/train/doc_master.ndjson
train_query_master: ${.DATA_PATH}/train/query_master.ndjson
train_positives: ${.DATA_PATH}/train/positive_lists.ndjson

validation_doc_master: ${.DATA_PATH}/validation/doc_master.ndjson
validation_query_master: ${.DATA_PATH}/validation/query_master.ndjson
validation_positives: ${.DATA_PATH}/validation/positive_lists.ndjson

hard_negative_scores: ${.DATA_PATH}/hard-negatives-cross-encoder-scores.ndjson.gz
```

## Sampling Modes

The `TripletDistilDataset` supports different sampling strategies:

### Query-Based Sampling (Default, Currently Supported)

- **Dataset size**: Equals the number of queries
- **Behavior**: For each epoch iteration:
  1. One sample corresponds to one query
  2. Positive document is **randomly sampled** from the query's positive list
  3. Negative document is **randomly sampled** from hard negatives (documents with scores but not in positive list)
- **Training recommendation**: Use **multi-epoch training** to ensure all positive pairs are utilized over time
- **Advantage**: Balanced training across all queries

### Positive-Pair-Based Sampling (Future Support)

- **Dataset size**: Equals the total number of positive pairs across all queries
- **Behavior**: One sample corresponds to one (query, positive document) pair
- **Training recommendation**: Single epoch covers all positive pairs
- **Advantage**: Ensures all positive pairs are seen in one epoch

## Data Validation Rules

The `TripletDistilDataset` class enforces the following validation rules:

1. **Query Coverage**: Every query ID in the positive list must exist in the query master
2. **Query Completeness**: Every query ID in the query master must have entries in the positive list
3. **Document Existence**: All document IDs referenced in the positive lists must exist in the document master
4. **Positive Requirement**: Every query must have at least one positive document
5. **Negative Availability**: Every query must have at least one available negative document (in scores but not in positive list and exists in document master)
6. **Positive Score Requirement**: Every (query, positive document) pair must have a teacher score in the hard negative scores file

If any validation rule fails, the system will raise a `ValueError` with a specific error message including the problematic ID for easy debugging.

## Dataset Output

When loading a sample from the `TripletDistilDataset`, you receive a **5-tuple**:

```python
(query_text, positive_doc_text, negative_doc_text, positive_score, negative_score)
```

**Fields**

1. `query_text` (str): The query text
2. `positive_doc_text` (str): The positive (relevant) document text
3. `negative_doc_text` (str): The negative (hard negative) document text
4. `positive_score` (float): Teacher model's similarity score for (query, positive_doc) pair
5. `negative_score` (float): Teacher model's similarity score for (query, negative_doc) pair

**Example**

```python
(
  "Gitでコミットを取り消すにはどうすればよい？",
  "直前のコミットを取り消すにはgit reset --softやgit revertを状況に応じて使い分けます。",
  "雨天時は運動会が体育館で実施される予定です。",
  0.95,
  0.23
)
```

## Use Case

This data format is designed for **SPLADE training with knowledge distillation**, where:

1. **Knowledge Transfer**: The student model (SPLADE) learns from a strong teacher model (typically a cross-encoder)
2. **Hard Negative Mining**: Negative documents are sampled from candidates that the teacher rates highly but are not actually relevant
3. **Soft Labels**: Teacher scores provide soft supervision signals in addition to hard labels (positive/negative)

The model learns to

- Score queries and documents in a sparse lexical space
- Mimic the teacher model's scoring behavior through distillation loss
- Rank positive documents higher than negative documents
- Learn from challenging hard negatives
- Produce sparse, interpretable representations

According to the [SPLADE v2bis](https://arxiv.org/pdf/2010.02666) paper, distillation training achieves higher accuracy than training without distillation (e.g., the triplet-based training described in [splade_triplet_data_format.md](splade_triplet_data_format.md)).

## Generating Hard Negative Scores

To create the hard negative scores file, you typically

1. **Select a Teacher Model**: Use a strong cross-encoder
2. **Generate Candidate Pool**: For each query, retrieve top-k documents using BM25 or another retrieval method (e.g., k=100-1000). Also add the positive documents to the candidate pool.
3. **Score Pairs**: Use the teacher model to score all (query, candidate_document) pairs
4. **Save Scores**: Store scores in the required NDJSON format


## Reference

- Implementation: [triplet_distil_dataset.py](../src/light_splade/data/triplet_distil_dataset.py)
- Score loader: [pair_score.py](../src/light_splade/data/pair_score.py)
- Data schemas: [schemas/data/__init__.py](../src/light_splade/schemas/data/__init__.py)
- Example config: [splade_data_mmarco_ja_distill.yaml](../config/data/splade_data_mmarco_ja_distill.yaml)
