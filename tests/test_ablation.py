"""CPU test of the training-budget ablation with the tiny in-code model (no downloads)."""

import json

from clinical_lora.ablation import AblationConfig, config_name, run_ablation, select_best
from clinical_lora.cli import main
from clinical_lora.testing import make_tiny_model_and_tokenizer, synthetic_examples


def _row(name, f1, cost):
    return {"name": name, "val_strict_f1": {"value": f1}, "cost_units": cost}


def test_select_best_prefers_higher_f1_then_cheaper_run():
    rows = [_row("a", 0.80, 2000), _row("b", 0.85, 6000), _row("c", 0.85, 4000)]
    assert select_best(rows)["name"] == "c"
    assert config_name(None, 2.0) == "n=all, epochs=2" and config_name(2000, 1.0).endswith("=1")


def test_ablation_end_to_end_on_tiny_model(tmp_path):
    _, tok = make_tiny_model_and_tokenizer()
    ex = synthetic_examples(8)
    splits = {"train": ex, "validation": ex[:4], "test": ex[4:8]}
    trained = []

    def factory():
        model, _ = make_tiny_model_and_tokenizer()
        trained.append(model)
        return model

    def load_adapter(base, path):
        from peft import PeftModel

        return PeftModel.from_pretrained(base, path)

    cfg = AblationConfig(
        base_model="tiny",
        epochs=(1.0, 2.0),
        n_train=(4, None),
        reference=(4, 1.0),
        n_val=4,
        n_test=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_length=2048,
        max_new_tokens=8,
        eval_batch_size=2,
        n_boot=20,
        out_dir=str(tmp_path / "res"),
        adapter_root=str(tmp_path / "ad"),
    )
    res = run_ablation(
        cfg, model_factory=factory, tokenizer=tok, splits=splits, load_adapter=load_adapter
    )

    names = [r["name"] for r in res["grid"]]
    assert names == ["n=4, epochs=1", "n=4, epochs=2", "n=all, epochs=1", "n=all, epochs=2"]
    assert [r["n_train"] for r in res["grid"]] == [4, 4, 8, 8]
    assert res["reference"] == "n=4, epochs=1" and res["selected"] in names
    # each configuration starts from a fresh base model, plus one per test evaluation
    n_test_evals = 1 if res["selected"] == res["reference"] else 2
    assert len(trained) == len(names) + n_test_evals
    assert set(res["test"]["systems"]) == {res["reference"], res["selected"]}
    assert res["test"]["meta"]["n_test"] == 4
    saved = json.loads((tmp_path / "res" / "ablation.json").read_text())
    assert saved["selected"] == res["selected"]
    md = (tmp_path / "res" / "ablation.md").read_text()
    assert "(reference)" in md and "Val strict F1" in md
    assert (tmp_path / "res" / "ablation_val_f1.png").stat().st_size > 1000


def test_cli_ablate_parses_sizes(monkeypatch):
    seen = {}

    def fake_run(cfg):
        seen["cfg"] = cfg
        return {"selected": "x", "reference": "y"}

    monkeypatch.setattr("clinical_lora.ablation.run_ablation", fake_run)
    main(["ablate", "--epochs", "1", "3", "--n-train", "500", "all", "--n-val", "50"])
    cfg = seen["cfg"]
    assert cfg.epochs == (1.0, 3.0) and cfg.n_train == (500, None) and cfg.n_val == 50
