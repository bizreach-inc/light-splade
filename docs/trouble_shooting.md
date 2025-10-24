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

# Running the training script on CPU-only machines

With the current default settings, running the training script on a CPU-only machine can raise the following error:

```text
ValueError: fp16 mixed precision requires a GPU (not 'mps').
```

This is caused by an issue in the `accelerate` package that has been fixed in [this Pull Request](https://github.com/huggingface/accelerate/pull/3373)

Update `accelerate` to the latest version to get code of the above PR and (optionally) upgrade `torch` to the required version as shown below.

```bash
uv remove accelerate
uv add git+https://github.com/huggingface/accelerate.git
uv add torch==2.8.0
```

After updating, the training script should run on CPU-only environments without the fp16/GPU error.

**NOTE**: Running the training script on CPU-only machines is intended for demonstration purposes only. For full training runs, it is recommended to use a GPU-enabled machine.
