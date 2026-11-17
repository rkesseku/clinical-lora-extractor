# Evaluation report

Held-out test set: **200** documents (identical for every system). 95% percentile-bootstrap CIs over documents (1000 resamples).
- base_model: `Qwen/Qwen2.5-1.5B-Instruct`
- dataset: `BC5CDR (biomedical literature; proxy for clinical text, no PHI)`
- n_train: `2000`
- lora: `{'r': 16, 'alpha': 32, '4bit': False}`

## Entity-level F1 (micro) with 95% CI

### Strict match

| System | Chemicals F1 | Diseases F1 | Overall P | Overall R | Overall F1 |
|---|---|---|---|---|---|
| base few-shot (k=3) | 0.551 [0.453, 0.636] | 0.348 [0.265, 0.425] | 0.665 | 0.352 | 0.460 [0.392, 0.521] |
| LoRA fine-tuned (0-shot) | 0.846 [0.793, 0.899] | 0.795 [0.737, 0.849] | 0.859 | 0.790 | 0.823 [0.784, 0.861] |

### Relaxed match

| System | Chemicals F1 | Diseases F1 | Overall P | Overall R | Overall F1 |
|---|---|---|---|---|---|
| base few-shot (k=3) | 0.622 [0.529, 0.710] | 0.470 [0.381, 0.551] | 0.799 | 0.424 | 0.554 [0.492, 0.617] |
| LoRA fine-tuned (0-shot) | 0.860 [0.809, 0.911] | 0.863 [0.813, 0.905] | 0.897 | 0.828 | 0.861 [0.823, 0.896] |

## Output validity, latency, tokens, cost

| System | Strict-JSON rate | Parsed after repair | Latency mean / p95 (s) | Prompt tok | New tok | Cost / 1k ex (USD) |
|---|---|---|---|---|---|---|
| base few-shot (k=3) | 1.000 | 1.000 | 0.237 / 0.447 | 381 | 17 | n/a |
| LoRA fine-tuned (0-shot) | 1.000 | 1.000 | 0.406 / 0.609 | 113 | 20 | n/a |

## Paired bootstrap: overall F1 of `LoRA fine-tuned (0-shot)` minus `base few-shot (k=3)`

- strict: diff = +0.362, 95% CI [+0.308, +0.423], two-sided bootstrap p = 0.000
- relaxed: diff = +0.307, 95% CI [+0.256, +0.359], two-sided bootstrap p = 0.000

## Notes

- Invalid/unparseable outputs are scored as empty predictions (all gold entities missed).
- Latency is wall-clock generation time divided by batch size (amortised); compare systems only when run on the same hardware and batch size.
- Local models have no per-token cost; `n/a` means no price was supplied.
