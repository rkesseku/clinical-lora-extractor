"""Command line entry point: ``clinical-lora run`` (end-to-end) and ``clinical-lora serve``."""

from __future__ import annotations

import argparse
import json
import os

from .pipeline import ExperimentConfig, run_experiment


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clinical-lora")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="few-shot baseline + LoRA training + evaluation (needs a GPU)")
    r.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    r.add_argument("--n-train", type=int, default=2000)
    r.add_argument("--n-test", type=int, default=200)
    r.add_argument("--n-shots", type=int, default=3)
    r.add_argument("--epochs", type=float, default=1.0)
    r.add_argument("--max-steps", type=int, default=-1)
    r.add_argument("--load-in-4bit", action="store_true")
    r.add_argument("--api", action="store_true", help="also evaluate the hosted API baseline")
    r.add_argument("--price-in", type=float, default=None, help="API USD per 1M input tokens")
    r.add_argument("--price-out", type=float, default=None, help="API USD per 1M output tokens")
    r.add_argument("--out", default="results")
    r.add_argument("--adapter-dir", default="adapters/clinical-lora")

    s = sub.add_parser("serve", help="serve base model + adapter over FastAPI")
    s.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    s.add_argument("--adapter", default=None)
    s.add_argument("--merge", action="store_true")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        cfg = ExperimentConfig(
            base_model=args.base_model,
            n_train=args.n_train,
            n_test=args.n_test,
            n_shots=args.n_shots,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            load_in_4bit=args.load_in_4bit,
            use_api=args.api,
            api_price_in=args.price_in,
            api_price_out=args.price_out,
            out_dir=args.out,
            adapter_dir=args.adapter_dir,
        )
        res = run_experiment(cfg)
        print(json.dumps(res["meta"], indent=2))
    else:
        import uvicorn  # optional extra: pip install "clinical-lora-extractor[serve]"

        os.environ["CLE_BASE_MODEL"] = args.base_model
        if args.adapter:
            os.environ["CLE_ADAPTER"] = args.adapter
        os.environ["CLE_MERGE"] = "1" if args.merge else "0"
        uvicorn.run(
            "clinical_lora.serve:create_app_from_env", factory=True, host=args.host, port=args.port
        )


if __name__ == "__main__":
    main()
