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

import pytest

from light_splade.regularizer import RegularizerScheduler


class TestRegularizerScheduler:
    def test_initialization(self) -> None:
        scheduler = RegularizerScheduler(lambda_=0.1, T=100)
        assert scheduler._initial_lambda == 0.1
        assert scheduler._warmup_steps == 100
        assert scheduler._curr_step == 0
        assert scheduler._lambda == 0

    def test_get_lambda_at_start(self) -> None:
        scheduler = RegularizerScheduler(lambda_=0.5, T=50)
        assert scheduler.get_lambda() == 0.0

    def test_step_increases_lambda_quadratically_within_warmup(self) -> None:
        initial_lambda = 1.0
        warmup_steps = 10
        scheduler = RegularizerScheduler(lambda_=initial_lambda, T=warmup_steps)

        expected_lambdas = [
            min(initial_lambda, initial_lambda * (1 / warmup_steps) ** 2),
            min(initial_lambda, initial_lambda * (2 / warmup_steps) ** 2),
            min(initial_lambda, initial_lambda * (3 / warmup_steps) ** 2),
            min(initial_lambda, initial_lambda * (4 / warmup_steps) ** 2),
        ]

        for i in range(1, 5):
            assert pytest.approx(scheduler.step()) == expected_lambdas[i - 1]
            assert scheduler._curr_step == i

    def test_get_lambda_increases_quadratically_within_warmup(self) -> None:
        initial_lambda = 0.8
        warmup_steps = 5
        scheduler = RegularizerScheduler(lambda_=initial_lambda, T=warmup_steps)

        expected_lambdas = [
            0.0,
            min(initial_lambda, initial_lambda * (1 / warmup_steps) ** 2),
            min(initial_lambda, initial_lambda * (2 / warmup_steps) ** 2),
            min(initial_lambda, initial_lambda * (3 / warmup_steps) ** 2),
            min(initial_lambda, initial_lambda * (4 / warmup_steps) ** 2),
            initial_lambda,
        ]

        for i in range(warmup_steps + 1):
            assert pytest.approx(scheduler.get_lambda()) == expected_lambdas[i]
            if i < warmup_steps:
                scheduler.step()

    def test_step_returns_initial_lambda_after_warmup(self) -> None:
        initial_lambda = 0.3
        warmup_steps = 3
        scheduler = RegularizerScheduler(lambda_=initial_lambda, T=warmup_steps)

        for _ in range(warmup_steps):
            scheduler.step()
        for _ in range(5):
            assert pytest.approx(scheduler.step()) == initial_lambda
            assert scheduler._curr_step == warmup_steps

    def test_get_lambda_returns_initial_lambda_after_warmup(self) -> None:
        initial_lambda = 0.6
        warmup_steps = 7
        scheduler = RegularizerScheduler(lambda_=initial_lambda, T=warmup_steps)

        for _ in range(warmup_steps + 2):
            scheduler.step()
        assert pytest.approx(scheduler.get_lambda()) == initial_lambda
        assert scheduler._curr_step == warmup_steps

    def test_set_step_updates_lambda(self) -> None:
        initial_lambda = 0.9
        warmup_steps = 10
        scheduler = RegularizerScheduler(lambda_=initial_lambda, T=warmup_steps)

        scheduler.set_step(3)
        expected_lambda = min(initial_lambda, initial_lambda * (3 / warmup_steps) ** 2)
        assert pytest.approx(scheduler.get_lambda()) == expected_lambda
        assert scheduler._curr_step == 3

        scheduler.set_step(warmup_steps + 2)
        assert pytest.approx(scheduler.get_lambda()) == initial_lambda
        assert scheduler._curr_step == warmup_steps + 2

        scheduler.set_step(0)
        assert pytest.approx(scheduler.get_lambda()) == 0.0
        assert scheduler._curr_step == 0
