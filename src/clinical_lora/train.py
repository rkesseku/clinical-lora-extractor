"""LoRA (PEFT) training with the plain Hugging Face ``Trainer``.

Loss is computed only on the JSON answer tokens (prompt tokens are masked with -100).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .data import Example
from .prompts import build_prompt

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class TrainConfig:
    output_dir: str = "adapters/clinical-lora"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    gradient_checkpointing: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)


def encode_example(tokenizer, ex: Example, max_length: int) -> dict[str, list[int]]:
    """Tokenise ``prompt + JSON answer + EOS``; labels are -100 on the prompt."""
    prompt_ids = tokenizer(build_prompt(tokenizer, ex.text), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(ex.target_json(), add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    # Truncate the prompt (left side of the document) rather than the answer.
    room = max_length - len(answer_ids)
    if room < 1:
        raise ValueError("max_length too small for the answer")
    prompt_ids = prompt_ids[-room:] if len(prompt_ids) > room else prompt_ids
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples: list[Example], max_length: int):
        self.rows = [encode_example(tokenizer, e, max_length) for e in examples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        for i, b in enumerate(batch):  # right padding for training
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            att[i, :k] = 1
            lab[i, :k] = torch.tensor(b["labels"])
        return {"input_ids": ids, "attention_mask": att, "labels": lab}


def attach_lora(model, cfg: TrainConfig, quantized: bool = False):
    """Wrap ``model`` with a LoRA adapter; trainable params stay fp32 (fp16 AMP)."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def train_lora(model, tokenizer, train_examples: list[Example], cfg: TrainConfig, quantized=False):
    """Fine-tune ``model`` with LoRA on ``train_examples``; saves the adapter to ``cfg.output_dir``.

    Returns ``(peft_model, train_metrics)``.
    """
    from transformers import Trainer, TrainingArguments

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = attach_lora(model, cfg, quantized=quantized)
    dataset = SFTDataset(tokenizer, train_examples, cfg.max_length)
    use_cuda = torch.cuda.is_available()
    kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        report_to="none",
        fp16=use_cuda,  # T4 has no bf16
        seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=use_cuda,
        **cfg.extra,
    )
    try:  # transformers 4.x
        args = TrainingArguments(warmup_ratio=cfg.warmup_ratio, **kwargs)
    except TypeError:  # transformers 5.x: a float < 1 in `warmup_steps` is a ratio
        args = TrainingArguments(warmup_steps=cfg.warmup_ratio, **kwargs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    out = trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return model, {**out.metrics, "n_train": len(dataset)}
