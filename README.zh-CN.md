# VLM-SFT

[English](README.md) | [简体中文](README.zh-CN.md)

VLM-SFT 是一个使用 TRL 和 Transformers 编写的视觉语言模型监督微调示例。核心脚本保持短小直观，方便根据不同 VLM checkpoint 和 JSON 格式对话数据集进行改造。

## 功能

- 通过 `AutoProcessor` 和 `AutoModelForCausalLM` 加载本地 VLM checkpoint
- 对多模态对话数据应用模型 chat template
- mask prompt token，让 loss 主要作用在 assistant 输出上
- 冻结 vision tower 和 multimodal projector
- 使用 `SFTTrainer` 微调语言侧参数

## 文件

```text
base.py      # 最小化 TRL + Transformers VLM SFT 脚本
```

## 数据格式

脚本默认读取 `hf_format_data.json`。每条样本需要包含：

- `conversations`：与所选 processor 兼容的 chat 风格消息
- `images`：样本引用的图片路径

如果数据集路径不同，请修改 `base.py` 中的 `data_files`。

## 模型路径

`base.py` 当前使用：

```python
model_path = "/mnt/kimi_vl"
```

运行前请改成本地 VLM checkpoint 路径。

## 安装

创建 Python 环境并安装常用依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install torch transformers datasets pillow numpy trl
```

如果模型和 GPU 环境需要 `attn_implementation="flash_attention_2"`，请单独安装 FlashAttention。

## 运行

```bash
python base.py
```

正式训练前，请先检查 `base.py` 中的硬编码路径和训练参数。

## 说明

这是一个最小训练模板，不是完整训练框架。建议把它作为起点，根据自己的数据集和模型调整预处理、collator、模型路径和 SFT 配置。

## 许可证

本项目使用 Apache License 2.0，见 [LICENSE](LICENSE)。
