"""CPU smoke test: tiny random Llama + in-code tokenizer. Nothing is downloaded."""

import json

import pytest
import torch

from clinical_lora.evaluate import SystemRun, build_results, render_markdown, write_outputs
from clinical_lora.inference import Generator, Prediction, predict
from clinical_lora.parsing import parse_extraction
from clinical_lora.pipeline import ExperimentConfig, run_experiment
from clinical_lora.serve import Extractor, create_app, hf_text_generator
from clinical_lora.testing import make_tiny_model_and_tokenizer, synthetic_examples
from clinical_lora.train import TrainConfig, encode_example, train_lora


@pytest.fixture(scope="module")
def tiny():
    return make_tiny_model_and_tokenizer()


def test_tokenizer_round_trips_json():
    _, tok = make_tiny_model_and_tokenizer()
    s = '{"chemicals": ["Naloxone"], "diseases": ["hypertension"]}'
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == s


def test_label_masking_only_answer_tokens(tiny):
    _, tok = tiny
    ex = synthetic_examples(1)[0]
    row = encode_example(tok, ex, max_length=2048)
    answer = [t for t in row["labels"] if t != -100]
    assert answer[-1] == tok.eos_token_id
    assert tok.decode(answer[:-1]) == ex.target_json()
    assert len(row["input_ids"]) == len(row["labels"])


def test_lora_training_changes_only_adapter_and_loss_drops(tmp_path):
    model, tok = make_tiny_model_and_tokenizer()
    before = {n: p.clone() for n, p in model.named_parameters()}
    cfg = TrainConfig(
        output_dir=str(tmp_path / "ad"),
        r=4,
        lora_alpha=8,
        max_steps=25,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-3,
        logging_steps=5,
        max_length=1024,
        target_modules=("q_proj", "v_proj"),
    )
    pm, metrics = train_lora(model, tok, synthetic_examples(8), cfg)
    assert metrics["n_train"] == 8 and (tmp_path / "ad" / "adapter_config.json").exists()
    # base weights frozen, adapter trained
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    for n, p in pm.named_parameters():
        key = n.replace("base_model.model.", "").replace(".base_layer", "")
        if "lora_" not in n and key in before:
            assert torch.equal(p, before[key]), n
    assert metrics["train_loss"] < 6.0  # sanity: finite and well below ln(vocab) * few


def test_generation_and_prediction_records(tiny):
    model, tok = tiny
    gen = Generator(model, tok, max_new_tokens=12, batch_size=2)
    preds = predict(gen, synthetic_examples(3))
    assert len(preds) == 3
    for p in preds:
        assert isinstance(p.raw, str) and 0 < p.new_tokens <= 12 and p.prompt_tokens > 0
        assert p.latency_s > 0


def test_overfit_tiny_model_then_extract_valid_json(tmp_path):
    """A few dozen steps on 4 docs is enough for the tiny model to memorise the format."""
    model, tok = make_tiny_model_and_tokenizer(hidden=64)
    ex = synthetic_examples(4)
    cfg = TrainConfig(
        output_dir=str(tmp_path / "ad"),
        r=16,
        lora_alpha=32,
        max_steps=150,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-2,
        logging_steps=50,
        lora_dropout=0.0,
    )
    pm, _ = train_lora(model, tok, ex, cfg)
    preds = predict(Generator(pm, tok, max_new_tokens=60, batch_size=4), ex)
    assert sum(p.parse.valid for p in preds) >= 3
    golds = [e.entities for e in ex]
    res = build_results([SystemRun("lora", preds)], ex, n_boot=50)
    assert res["systems"]["lora"]["entity"]["strict"]["overall"]["f1"]["value"] > 0.5
    # Serving path with the real tiny model
    client_gen = hf_text_generator(pm, tok, max_new_tokens=60)
    ext = Extractor(client_gen, tok, max_attempts=2)
    from fastapi.testclient import TestClient

    r = TestClient(create_app(ext)).post("/extract", json={"text": ex[0].text})
    assert r.status_code == 200 and r.json()["extraction"] == golds[0]


def _fake_preds(examples, drop_diseases=False, invalid_every=0):
    out = []
    for i, e in enumerate(examples):
        d = {k: list(v) for k, v in e.entities.items()}
        if drop_diseases:
            d["diseases"] = []
        raw = json.dumps(d)
        if invalid_every and i % invalid_every == 0:
            raw = "sorry, cannot do that"
        out.append(Prediction(raw, parse_extraction(raw), 0.1 + 0.01 * i, 100, 20))
    return out


def test_full_evaluation_and_report_on_synthetic_examples(tmp_path):
    test = synthetic_examples(12)
    runs = [
        SystemRun("weak", _fake_preds(test, drop_diseases=True, invalid_every=3)),
        SystemRun("api", _fake_preds(test), price_per_mtok=(1.0, 5.0), note="stub"),
        SystemRun("strong", _fake_preds(test)),
    ]
    res = build_results(runs, test, meta={"base_model": "tiny"}, n_boot=100)
    s = res["systems"]
    assert s["strong"]["entity"]["strict"]["overall"]["f1"]["value"] == pytest.approx(1.0)
    assert s["weak"]["entity"]["strict"]["diseases"]["recall"]["value"] == 0.0
    assert s["weak"]["validity"]["strict_json_rate"] == pytest.approx(8 / 12)
    assert s["api"]["cost_usd"]["total"] == pytest.approx((12 * 100 * 1.0 + 12 * 20 * 5.0) / 1e6)
    assert s["strong"]["cost_usd"] is None
    assert res["paired_f1_difference"]["strict"]["diff"] > 0
    paths = write_outputs(res, tmp_path)
    assert json.loads(paths["metrics"].read_text())["meta"]["n_test"] == 12
    assert paths["f1_by_type"].stat().st_size > 1000 and paths["latency_validity"].exists()
    md = paths["report"].read_text()
    assert "Strict match" in md and "weak" in md and "Paired bootstrap" in md
    assert render_markdown(res) == md


def test_pipeline_end_to_end_with_tiny_model(tmp_path):
    model, tok = make_tiny_model_and_tokenizer()
    ex = synthetic_examples(8)
    splits = {"train": ex, "validation": ex[:2], "test": ex[:4]}
    cfg = ExperimentConfig(
        n_train=8,
        n_test=4,
        n_shots=1,
        max_steps=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_new_tokens=8,
        eval_batch_size=2,
        n_boot=20,
        out_dir=str(tmp_path / "res"),
        adapter_dir=str(tmp_path / "ad"),
        base_model="tiny",
        max_length=2048,
    )
    res = run_experiment(cfg, model=model, tokenizer=tok, splits=splits)
    assert list(res["systems"]) == ["base few-shot (k=1)", "LoRA fine-tuned (0-shot)"]
    assert (tmp_path / "res" / "metrics.json").exists()
    assert (tmp_path / "res" / "report.md").exists()
    assert (tmp_path / "res" / "predictions.jsonl").exists()
