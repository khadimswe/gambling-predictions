"""Command line entry point: `soccer <command>` (installed by `pip install -e .`)."""

from __future__ import annotations

import argparse
import json
import sys

from soccer_predict import config


def cmd_download(args: argparse.Namespace) -> None:
    from soccer_predict import data

    paths = data.download(force=args.force)
    print(f"{len(paths)} raw files in {config.RAW_DIR}")


def cmd_build(args: argparse.Namespace) -> None:
    from soccer_predict import pipeline

    out = pipeline.build(n_tiers=args.tiers)
    counts = out.features["split"].value_counts().to_dict()
    print(f"wrote {config.FEATURES_CSV} ({len(out.features)} matches: {counts})")


def cmd_tiers_report(args: argparse.Namespace) -> None:
    from soccer_predict import tiers_report

    summary = tiers_report.run(example_match_id=args.example)
    print(json.dumps(summary, indent=2, default=float))
    print(f"results in {config.RESULTS_DIR}, figures in {config.FIGURES_DIR}")


def cmd_train(args: argparse.Namespace) -> None:
    from soccer_predict import classifier, evaluate, pipeline

    df = pipeline.load_features()
    use_tiers = not args.no_tiers
    train = df[df["split"] == "train"]
    model = classifier.train(train, kind=args.model, use_tiers=use_tiers)

    results = {"model": args.model, "use_tiers": use_tiers}
    splits = ["val", "test"] if args.test else ["val"]
    for split in splits:
        part = df[df["split"] == split]
        pred, proba = classifier.predict(model, part, use_tiers=use_tiers)
        results[split] = evaluate.evaluate(part["target"], pred, proba)

    name = f"classifier_{args.model}{'' if use_tiers else '_no_tiers'}.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / name).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="soccer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch raw CSVs from football-data.co.uk")
    p.add_argument("--force", action="store_true", help="re-download files that exist")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("build", help="clean data, compute form, fit tiers, write features.csv")
    p.add_argument("--tiers", type=int, default=config.N_TIERS, help="number of tiers (K)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("tiers-report", help="run Khadim's tier experiments and write figures")
    p.add_argument("--example", default="E0_1516_20160206_ManCity", help="match_id to trace")
    p.set_defaults(func=cmd_tiers_report)

    p = sub.add_parser("train", help="train and evaluate Ryann's classifier")
    p.add_argument("--model", default="logreg", help="model kind passed to make_model")
    p.add_argument("--no-tiers", action="store_true", help="ablation: drop tier features")
    p.add_argument("--test", action="store_true", help="also score the test season (do this once)")
    p.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, NotImplementedError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
