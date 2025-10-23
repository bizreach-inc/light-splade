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

"""Regularizer scheduling utilities.

This module implements a small scheduler used to progressively increase the strength of a regularizer's lambda value
with a quadratic warmup until it reaches a predefined maximum. The implementation follows the form used in several
SPLADE-related papers where regularization strength is ramped up to encourage stable training.

Reference:
    Biswajit Paria et al., "Minimizing FLOPs to Learn Efficient Sparse
    Representations" (https://arxiv.org/abs/2004.05665)
"""


class RegularizerScheduler:
    """Scheduler that ramps the regularizer lambda from 0 to ``lambda_``.

    The scheduler performs a quadratic warmup over ``T`` steps. After the warmup period the lambda remains fixed at the
    initial value provided at construction.

    Args:
        lambda_ (float): Target lambda value after warmup.
        T (int): Number of warmup steps.
    """

    def __init__(self, lambda_: float, T: int) -> None:
        self._initial_lambda: float = lambda_
        self._warmup_steps: int = T
        self._curr_step: int = 0
        self._lambda: float = 0.0

    def _recompute_lambda(self) -> None:
        """Recompute the current lambda value based on the current step."""
        self._lambda = min(
            self._initial_lambda,
            self._initial_lambda * ((self._curr_step / self._warmup_steps) ** 2),
        )

    def set_step(self, step: int) -> None:
        """Set the internal step counter and recompute the lambda.

        Args:
            step (int): Absolute step number to set the scheduler to.
        """
        self._curr_step = step
        self._recompute_lambda()

    def step(self) -> float:
        """Advance the scheduler by one step and return the current lambda.

        Returns:
            float: Current lambda value after stepping.
        """
        if self._curr_step >= self._warmup_steps:
            return self._lambda

        self._curr_step += 1
        self._recompute_lambda()
        return self._lambda

    def get_lambda(self) -> float:
        """Return the current lambda value without changing state."""
        return self._lambda
