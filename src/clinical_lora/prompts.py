"""Prompt templates. The same instruction is used for few-shot prompting and LoRA training."""

from __future__ import annotations

from .data import Example

SYSTEM_PROMPT = (
    "You are a biomedical information extraction system. Extract every chemical/drug mention and "
    "every disease mention from the text. Reply with ONLY a JSON object of the form "
    '{"chemicals": [...], "diseases": [...]}. Copy mentions exactly as written in the text, '
    "list each distinct mention once, and use empty lists when there are none."
)


def user_message(text: str) -> str:
    return f"Text:\n{text}\n\nJSON:"


def select_shots(train: list[Example], k: int, max_chars: int = 900) -> list[Example]:
    """Pick ``k`` deterministic few-shot demonstrations: short docs with both entity types."""
    if k <= 0:
        return []
    good = [
        e
        for e in train
        if len(e.text) <= max_chars and e.entities.get("chemicals") and e.entities.get("diseases")
    ]
    pool = good if len(good) >= k else train
    # Evenly spaced picks keep it deterministic without favouring the head of the split.
    step = max(1, len(pool) // k)
    return [pool[i * step] for i in range(k)]


def build_messages(text: str, shots: list[Example] | None = None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots or []:
        msgs.append({"role": "user", "content": user_message(s.text)})
        msgs.append({"role": "assistant", "content": s.target_json()})
    msgs.append({"role": "user", "content": user_message(text)})
    return msgs


def build_prompt(tokenizer, text: str, shots: list[Example] | None = None) -> str:
    """Render the prompt string (ending where the model should start answering).

    Uses the tokenizer's chat template when it has one, else a plain-text fallback (used by the
    tiny test tokenizer). Tokenise the result with ``add_special_tokens=False``.
    """
    msgs = build_messages(text, shots)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    parts = [f"### System\n{msgs[0]['content']}\n"]
    for m in msgs[1:]:
        role = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"### {role}\n{m['content']}\n")
    parts.append("### Assistant\n")
    return "\n".join(parts)
