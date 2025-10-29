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

import torch


def contiguous(model: torch.nn.Module) -> torch.nn.Module:
    """Ensure module parameters are contiguous in memory.

    Safe serialization formats (e.g., safetensors) require parameter tensors to be contiguous. This helper iterates all
    parameters and replaces their data with a contiguous copy when necessary.

    Args:
        model (torch.nn.Module): Module whose parameters will be checked.

    Returns:
        torch.nn.Module: The same module instance with contiguous parameter data.
    """
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
    return model
