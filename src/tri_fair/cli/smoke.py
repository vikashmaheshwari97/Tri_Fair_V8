"""Run two search-split predictions to validate a real GPU/model backend before optimization."""

from __future__ import annotations
import argparse
import json
from promptolution.predictors import MarkerBasedPredictor
from tri_fair.config.datasets import ALL_DATASETS
from tri_fair.config.models import ALL_MODELS
from tri_fair.data.loading import create_dev_tasks
from tri_fair.data.partitions import split_search_validation
from tri_fair.runtime.models import create_llm
from tri_fair.runtime.decoding import configure_downstream_greedy
from tri_fair.cli.common import set_generation_limit


def run(args):
    config = ALL_DATASETS[args.dataset]
    parent, _ = create_dev_tasks(
        dataset_config=config,
        eval_strategy="full",
        seed=args.random_seed,
        manifest_dir=args.manifest_dir,
        regenerate_manifest=False,
    )
    search, _ = split_search_validation(parent)
    llm = create_llm(model_config=ALL_MODELS[args.model], seed=args.random_seed)
    configure_downstream_greedy(llm, seed=args.random_seed)
    set_generation_limit(llm, 16)
    predictor = MarkerBasedPredictor(llm, search.classes)
    xs = search.df[search.x_column].astype(str).iloc[:2].tolist()
    predictions, _ = predictor.predict([config.initial_prompts[0]] * len(xs), xs)
    if isinstance(predictions, str):
        predictions = [predictions]
    valid = len(predictions) == len(xs) and all(
        str(p).casefold() in search.classes for p in predictions
    )
    return {
        "model": args.model,
        "dataset": args.dataset,
        "search_examples": len(xs),
        "predictions": [str(p) for p in predictions],
        "allowed_labels": search.classes,
        "parsing_ok": valid,
        "downstream_tokens": llm.get_token_count(),
        "benchmark_result": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), default="qwen-3-30b")
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), default="bbq")
    parser.add_argument("--random-seed", type=int, default=72)
    parser.add_argument("--manifest-dir", default="data/splits_v8_clean")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["parsing_ok"] else 1)


if __name__ == "__main__":
    main()
