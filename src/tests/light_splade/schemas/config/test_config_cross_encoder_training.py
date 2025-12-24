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

from light_splade.schemas.config.base_cross_encoder import BaseConfigCrossEncoder


class TestBaseConfigCrossEncoder:
    def test_max_len(self) -> None:
        config = BaseConfigCrossEncoder(
            char_per_token_ratio=2.0,
            max_token_len=100
        )
        assert config.max_len == 200

    def test_not_allowed_max_len(self) -> None:
        with pytest.raises((TypeError)):
            config = BaseConfigCrossEncoder(
                char_per_token_ratio=2.0,
                max_token_len=100,
                max_len=300
            )

    def test_max_query_len(self) -> None:
        config = BaseConfigCrossEncoder(
            char_per_token_ratio=2.0,
            max_token_len=100,
            max_query_len=0.3
        )
        assert config.max_query_len == 60

    def test_invalid_max_query_len(self) -> None:
        with pytest.raises((ValueError)):
            config = BaseConfigCrossEncoder(
                char_per_token_ratio=2.0,
                max_token_len=100,
                max_query_len=1.1
            )

    def test_max_query_len_adjustment(self) -> None:
        config = BaseConfigCrossEncoder(
            char_per_token_ratio=2.0,
            max_token_len=100,
            max_query_len=1000
        )
        assert config.max_query_len == 200
