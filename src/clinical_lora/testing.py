"""Download-free tiny model + tokenizer for CPU smoke tests (random weights, built in code)."""

from __future__ import annotations

from .data import Example

SYNTHETIC_TEXTS = [
    "Naloxone reverses the antihypertensive effect of clonidine in patients with hypertension.",
    "Lidocaine caused seizures and cardiac arrest in a patient with renal failure.",
    "Aspirin reduces the risk of stroke but may cause gastric bleeding.",
    "Methotrexate induced hepatic toxicity and pneumonitis in two patients.",
]


def synthetic_examples(n: int = 8) -> list[Example]:
    """Deterministic fake documents with gold entities derived from the templates above."""
    gold = [
        {"chemicals": ["Naloxone", "clonidine"], "diseases": ["hypertension"]},
        {"chemicals": ["Lidocaine"], "diseases": ["seizures", "cardiac arrest", "renal failure"]},
        {"chemicals": ["Aspirin"], "diseases": ["stroke", "gastric bleeding"]},
        {"chemicals": ["Methotrexate"], "diseases": ["hepatic toxicity", "pneumonitis"]},
    ]
    return [
        Example(SYNTHETIC_TEXTS[i % 4], {k: list(v) for k, v in gold[i % 4].items()})
        for i in range(n)
    ]


def make_tiny_tokenizer(corpus: list[str] | None = None, vocab_size: int = 400):
    """Byte-level BPE trained in-process on a tiny corpus; no files, no network."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = corpus or (
        SYNTHETIC_TEXTS
        + ['{"chemicals": ["a"], "diseases": ["b"]}', "### System User Assistant Text JSON"]
    )
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tk.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>")


def make_tiny_model(tokenizer, hidden: int = 32, layers: int = 2, seed: int = 0):
    """Randomly initialised Llama-style causal LM sized for CPU tests."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return LlamaForCausalLM(cfg)


def make_tiny_model_and_tokenizer(**kw):
    tok = make_tiny_tokenizer()
    return make_tiny_model(tok, **kw), tok
