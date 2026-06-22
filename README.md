# VLM-SFT

[English](README.md) | [Simplified Chinese](README.zh-CN.md)

VLM-SFT is a compact vision-language model supervised fine-tuning example built
with TRL and Transformers. The core script keeps the training flow short and
explicit so it is easy to adapt to different VLM checkpoints and JSON-format
conversation datasets.

## What It Does

- Loads a local VLM checkpoint through `AutoProcessor` and
  `AutoModelForCausalLM`
- Applies the model chat template to multimodal conversation data
- Masks prompt tokens so the loss focuses on assistant output
- Freezes the vision tower and multimodal projector
- Fine-tunes the language side with `SFTTrainer`

## Files

```text
base.py      # Minimal TRL + Transformers VLM SFT script
```

## Data Format

The script expects a JSON dataset named `hf_format_data.json` by default. Each
sample should provide:

- `conversations`: chat-style messages compatible with the selected processor
- `images`: image paths referenced by the sample

Adjust `data_files` in `base.py` if your dataset uses another path.

## Model Path

`base.py` currently points to:

```python
model_path = "/mnt/kimi_vl"
```

Change this path to your local VLM checkpoint before running.

## Install

Create a Python environment and install the common dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install torch transformers datasets pillow numpy trl
```

Install FlashAttention separately if your model and GPU environment require
`attn_implementation="flash_attention_2"`.

## Run

```bash
python base.py
```

For real training, review the hard-coded paths and training arguments in
`base.py` first.

## Notes

This repository is a minimal training template, not a full training framework.
Use it as a starting point and adapt the preprocessing, collator, model path,
and SFT configuration for your dataset and model.

## License

Apache License 2.0. See [LICENSE](LICENSE).
