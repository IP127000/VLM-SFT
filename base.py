
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image
from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)
def mask_sequence_before_target_v2(lst, target_sequence):
    target_len = len(target_sequence)
    for i in range(len(lst) - target_len + 1):
        if lst[i:i+target_len] == target_sequence:
            end_index = i + target_len
            return [-100] * end_index + lst[end_index:]
    return lst

def preproces(examples):
    model_inputs = defaultdict(list)
    for i in range(len(examples["conversations"])):
        text = processor.apply_chat_template(examples["conversations"][i], add_generation_prompt=False)
        while processor.image_token in text:
            text = text.replace(
                processor.image_token,
                "<|placeholder|>" * 777,
                1,
            )
        text = text.replace("<|placeholder|>", processor.image_token)
        input_ids=processor.tokenizer.encode(text)
        labels=deepcopy(input_ids)
        labels = mask_sequence_before_target_v2(labels, [163588, 69702, 163601])
        input_ids=torch.from_numpy(np.array(input_ids))
        labels=torch.from_numpy(np.array(labels))
        ones_list = [1] * len(input_ids)
        attention_mask = np.array(ones_list)
        attention_mask=torch.from_numpy(attention_mask)
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["images"][i])
    return model_inputs

def collate_fn(examples):
    images = [Image.open(ex["images"][0])for ex in examples] 
    batch=processor.image_processor(images, return_tensors="pt") 
    batch["input_ids"]=torch.from_numpy(np.array([ex["input_ids"] for ex in examples]))
    batch["attention_mask"]=torch.from_numpy(np.array([ex["attention_mask"] for ex in examples]))
    batch["labels"]=torch.from_numpy(np.array([ex["labels"] for ex in examples]))
    return batch

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    training_args = SFTConfig(    
        output_dir="./cps",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        save_steps=500,
        learning_rate=1e-4,
        logging_steps=50,
        remove_unused_columns=False,
    )
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.max_seq_length=1000
    training_args.lr_scheduler_type="cosine"
    torch_dtype = (
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
        )
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
    )

    model_path="/mnt/kimi_vl"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,attn_implementation="flash_attention_2")

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    data_files="hf_format_data.json"
    dataset = load_dataset(path="json",data_files=data_files,split="train",num_proc=8,)
    column_names = list(next(iter(dataset)).keys())
    with training_args.main_process_first(desc="load dataset"):
        dataset = dataset.map(
            preproces,
            batched=True,
            batch_size=10,
            remove_columns=column_names,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )

    trainer.train()
