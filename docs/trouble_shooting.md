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
