# Copyright 2025 BizReach, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SPLADE model wrappers and utilities.

This module provides a lightweight PyTorch wrapper around transformer-based encoders to produce sparse lexical+expansion
representations as described in the SPLADE family of papers. It exposes two main classes:

- :class:`SpladeEncoder`: wraps a masked LM model and produces dense token activations that are aggregated and converted
to sparse dictionaries.
- :class:`Splade`: a convenience two-tower / siamese container that holds a document encoder and an optional separate
query encoder.

References:
[1]  SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval,
    - Thibault Formal, Benjamin Piwowarski, Carlos Lassance, and Stéphane Clinchant
    - https://arxiv.org/pdf/2109.10086

[2] From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models
    More Effective.
    - Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant.
    - SIGIR22 short paper (extension of SPLADE v2) (v2bis, SPLADE++)
    - https://arxiv.org/pdf/2205.04733

[3] An Efficiency Study for SPLADE Models.
    - Carlos Lassance and Stéphane Clinchant
    - SIGIR22 short paper (after v2bis, Efficient SPLADE)
    - https://arxiv.org/pdf/2207.03834
"""

import os
from logging import getLogger

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer

from light_splade.schemas.types import SPARSE_VECTOR_LIST
from light_splade.utils.model import contiguous

logger = getLogger(__name__)


QUERY_MODEL_FOLDER = "query_model"


def is_same_tokenizers(tokenizer1: PreTrainedTokenizer, tokenizer2: PreTrainedTokenizer) -> bool:
    """Return True when two tokenizers are of the same type and vocab.

    Args:
        tokenizer1 (PreTrainedTokenizer): First tokenizer instance.
        tokenizer2 (PreTrainedTokenizer): Second tokenizer instance.

    Returns:
        True if both tokenizers share the same class and vocabulary mapping.
    """
    return isinstance(tokenizer1, type(tokenizer2)) and tokenizer1.vocab == tokenizer2.vocab


class SpladeEncoder(torch.nn.Module):
    """Transformer-based encoder that emits aggregated token activations.

    The encoder loads a masked language model and its tokenizer. The forward pass returns a dense vector of
    vocabulary-sized activations which can be converted to a sparse dictionary representation by :meth:`to_sparse`.

    Args:
        model_path (str): Pretrained model path or identifier for the HF Auto classes.
        agg: Aggregation function over token positions, either ``"max"`` or ``"sum"``.
    """

    def __init__(
        self,
        model_path: str,
        agg: str = "max",
        device: str | None = None,
    ) -> None:
        assert agg in ["max", "sum"], f"agg must be one of ['max', 'sum'], but got {agg}"
        self.device: str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__()
        self.from_pretrained(model_path)
        self.agg_func = torch.max if agg == "max" else torch.sum

    def from_pretrained(self, model_path: str) -> None:
        """Load transformer model and tokenizer from ``model_path``."""
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # build mapping from index to token text
        self.idx2token = {idx: token for token, idx in self.tokenizer.get_vocab().items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and aggregate token activations.

        Args:
            input_ids (torch.Tensor): Token ids tensor of shape (batch, seq_len).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch, seq_len).

        Returns:
            Dense tensor of shape (batch, vocab_size) containing aggregated
            activations for each vocabulary term.
        """
        outs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = outs.logits  # (b, N, V)

        # Eq. (1) in [2]
        vecs, indices_ = self.agg_func(
            torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1),  # (b, N, 1)
            dim=1,
        )
        # vectors (b, V), which are weights `w_j` of the input query/doc over the vocab
        return vecs

    def to_sparse(self, dense: torch.Tensor) -> SPARSE_VECTOR_LIST:
        """Convert a single dense (vocab-sized) vector to a sparse dict.

        Note:
            Current implementation handles one vector at a time. The returned dictionary maps token strings to rounded
            float scores in descending order.

        Args:
            dense: 1D tensor of shape (V,) for a single vector or a 2D tensor for multi vectors.

        Returns:
            Mapping from token string to float score representing non-zero activations for the vector.
        """

        if len(dense.shape) == 2:
            return [self.to_sparse(dense[i])[0] for i in range(len(dense))]

        # TODO: check this code to support batch (this code currently supports only 1 vector) extract non-zero positions
        cols = dense.nonzero().squeeze().detach().cpu().tolist()
        if not isinstance(cols, list):
            cols = [cols]

        # extract the non-zero values
        weights = dense[cols].detach().cpu().tolist()

        # map token IDs to human-readable tokens and round scores for display
        sparse_dict_tokens_: dict[str, float] = {
            self.idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
        }
        # sort so most relevant tokens appear first
        sparse_vector: dict[str, float] = {
            k: v for k, v in sorted(sparse_dict_tokens_.items(), key=lambda item: item[1], reverse=True)
        }
        return [sparse_vector]

    def get_sparse(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SPARSE_VECTOR_LIST:
        """Return sparse representations for a batch of inputs.

        Args:
            input_ids (torch.Tensor): Tensor of shape (b, N).
            attention_mask (torch.Tensor): Tensor of shape (b, N).

        Returns:
            A list of sparse dictionaries, one per batch element.
        """
        embeddings = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        sparse_vecs = self.to_sparse(embeddings)
        return sparse_vecs

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        """Encode a list of sentences into embeddings (dense vectors)

        Args:
            sentences (list[str]): List of sentences to encode.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): Whether to show a progress bar during encoding.

        Returns:
            A tensor of shape (b, V) containing the encoded representations.
        """

        progress = tqdm(range(0, len(sentences), batch_size), disable=not show_progress_bar)

        embeddings_list = []
        for start in progress:
            end = min(start + batch_size, len(sentences))
            batch_sentences = sentences[start:end]
            token_outputs = self.tokenizer(batch_sentences, padding=True, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                embeddings_ = self.forward(
                    input_ids=token_outputs["input_ids"], attention_mask=token_outputs["attention_mask"]
                ).cpu()
                embeddings_list.append(embeddings_)

        embeddings = torch.vstack(embeddings_list)
        return embeddings


class Splade(torch.nn.Module):
    """Two-tower / siamese SPLADE model container.

    The :class:`Splade` container holds a document encoder and an optional separate query encoder. It provides
    convenience methods for forward encoding, converting dense outputs to sparse dictionaries, and saving / loading
    pretrained components.

    1. To create a siamese model (shared weight), let q_model_path be None
        ```python
        model = Splade(d_model_path=MODEL_PATH)
        ```

    2. To create a Two-Tower model, where 2 branches own 2 separate weights of the same model, set both d_model_path
    and q_model_path to the same model path.
        ```python
        model = Splade(
            d_model_path=MODEL_PATH,
            q_model_path=MODEL_PATH
        )
        ```

    3. To create a Two-Tower model, where 2 branches own 2 different models, set d_model_path and q_model_path to
    different model paths. Note that the 2 models must produce vectors with the same dimensionality.
        ```python
        model = Splade(
            d_model_path=MODEL_PATH_4_DOC,
            q_model_path=MODEL_PATH_4_QUERY
        )
        ```
    """

    def __init__(
        self,
        d_model_path: str,
        q_model_path: str | None = None,
        freeze_d_model: bool = False,
        agg: str = "max",
        device: str | None = None,
    ) -> None:
        """Create a Splade model.

        Args:
            d_model_path (str): Pretrained model path for the document encoder.
            q_model_path (str | None): Optional pretrained path for a separate query encoder.
            freeze_d_model (bool): If True, document encoder parameters will be frozen.
            agg: Aggregation method, either "max" or "sum".
        """
        if agg not in ["max", "sum"]:
            raise ValueError(f"agg must be one of ['max', 'sum'], but got {agg}")
        if freeze_d_model and q_model_path is None:
            raise ValueError(
                "Freezing the document encoder requires q_model_path to "
                "ensure the query encoder differs from the document encoder."
            )

        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.d_encoder = SpladeEncoder(d_model_path, agg=agg, device=device)
        self.q_encoder = self.d_encoder
        if q_model_path:
            self.q_encoder = SpladeEncoder(q_model_path, agg=agg, device=device)
        if freeze_d_model:
            self.d_encoder.requires_grad_(False)
        self.is_shared_weights = q_model_path is None

        self.d_model_path = d_model_path
        self.q_model_path = q_model_path
        self.agg = agg
        self.freeze_d_model = freeze_d_model

        if not is_same_tokenizers(self.d_encoder.tokenizer, self.q_encoder.tokenizer):
            raise ValueError("The two encoders have the same vocab size!")

    @property
    def num_trainable_params(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_params(self) -> int:
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        q_input_ids: torch.Tensor | None = None,
        q_attention_mask: torch.Tensor | None = None,
        d_input_ids: torch.Tensor | None = None,
        d_attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """Encode queries and/or documents and return dense vectors.

        The method accepts optional query and document inputs and returns a dictionary with keys ``q_vector`` and
        ``d_vector`` when available.
        """
        q_vec = None
        if q_input_ids is not None and q_attention_mask is not None:
            q_vec = self.q_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)

        d_vec = None
        if d_input_ids is not None and d_attention_mask is not None:
            d_vec = self.d_encoder(input_ids=d_input_ids, attention_mask=d_attention_mask)

        return dict(q_vector=q_vec, d_vector=d_vec)

    def to_sparse(self, denses: dict[str, torch.Tensor]) -> dict[str, SPARSE_VECTOR_LIST]:
        """Convert dense vectors produced by :meth:`forward` to sparse dicts.

        Args:
            denses (dict[str, torch.Tensor]): Mapping returned by :meth:`forward` containing ``q_vector`` and/or
                ``d_vector`` tensors.

        Returns:
            Mapping with keys ``q_vector`` and/or ``d_vector`` holding sparse dictionaries produced by
                :class:`SpladeEncoder.to_sparse`.
        """
        sparses = dict()
        if "q_vector" in denses:
            sparses["q_vector"] = self.q_encoder.to_sparse(denses["q_vector"])
        if "d_vector" in denses:
            sparses["d_vector"] = self.d_encoder.to_sparse(denses["d_vector"])
        return sparses

    def load(self, model_path: str) -> None:
        """Load encoders from a directory containing pretrained components.

        If the model uses separate query weights, the query encoder is expected to be in a subfolder named
        ``query_model``.
        """
        logger.info(f"Loading document encoder from {model_path}")
        self.d_encoder.from_pretrained(model_path)
        if not self.is_shared_weights:
            model_path_q = os.path.join(model_path, "query_model")
            logger.info(f"NOT Shared weight SPLADE -> Loading query encoder from {model_path_q}")
            self.q_encoder.from_pretrained(model_path_q)
        else:
            logger.info("Shared weight SPLADE -> Query encoder is the same as document encoder")
            self.q_encoder = self.d_encoder

        if self.freeze_d_model:
            self.d_encoder.requires_grad_(False)

        logger.info("--- After loading the model:")
        logger.info(f"{self.is_shared_weights=}")
        logger.info(f"{self.num_trainable_params=}")
        logger.info(f"{self.num_total_params=}")

    def save(self, output_dir: str, save_safetensors: bool = False) -> None:
        """Save encoder components to ``output_dir``.

        If ``save_safetensors`` is True, tensors are made contiguous before saving to enable safe serialization.
        """
        if save_safetensors:
            contiguous(self.d_encoder)
        self.d_encoder.transformer.save_pretrained(output_dir, safe_serialization=save_safetensors)
        self.d_encoder.tokenizer.save_pretrained(output_dir)

        if not self.is_shared_weights:
            output_dir_q = os.path.join(output_dir, QUERY_MODEL_FOLDER)
            os.makedirs(output_dir_q, exist_ok=True)

            if save_safetensors:
                contiguous(self.q_encoder)
            self.q_encoder.transformer.save_pretrained(output_dir_q, safe_serialization=save_safetensors)
            self.q_encoder.tokenizer.save_pretrained(output_dir_q)
