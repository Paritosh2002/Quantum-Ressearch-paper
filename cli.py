#!/usr/bin/env python
"""Unified command-line interface for the QML-on-CML benchmark.

Examples
--------
  python cli.py preprocess                         # download + build task .npz files
  python cli.py list-models                        # show available algorithms
  python cli.py train                              # train ALL models on ALL tasks
  python cli.py train --model vqc                  # one model, all tasks
  python cli.py train --model vqc qsvm --task diagnosis_binary
  python cli.py evaluate                           # print table + write markdown summary
  python cli.py visualize                          # render all figures
  python cli.py all                                # preprocess -> train -> evaluate -> visualize
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

from qmlcml import config as C
from qmlcml.models import list_models, REGISTRY


def _resolve_models(selected):
    if not selected or selected == ["all"]:
        return [name for name, _ in list_models()]
    unknown = [m for m in selected if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. "
                         f"Available: {sorted(REGISTRY)}")
    return selected


def _resolve_tasks(selected):
    if not selected or selected == ["all"]:
        return C.TASKS
    unknown = [t for t in selected if t not in C.TASKS]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Available: {C.TASKS}")
    return selected


def cmd_preprocess(_):
    from qmlcml import preprocess
    preprocess.main()


def cmd_list_models(_):
    print("Available models (representation -> name):")
    for name, cls in list_models():
        tag = "iterative" if getattr(cls, "iterative", False) else "one-shot"
        print(f"  [{cls.representation:9}] {name:20} ({tag})")


def cmd_train(args):
    from qmlcml import train
    models = _resolve_models(args.model)
    tasks = _resolve_tasks(args.task)
    kwargs = {}
    if args.qsvm_max_train is not None:
        kwargs["max_train"] = args.qsvm_max_train
    print(f"Training {models} on {tasks}")
    for task in tasks:
        for model in models:
            mk = kwargs if model == "qsvm" else {}
            train.run_model_on_task(model, task, **mk)
    print("\nDone. Next: python cli.py evaluate && python cli.py visualize")


def cmd_evaluate(_):
    from qmlcml import report
    report.print_table()
    report.write_markdown()


def cmd_visualize(_):
    from qmlcml import visualize
    visualize.all_figures()


def cmd_all(args):
    cmd_preprocess(args)
    cmd_train(args)
    cmd_evaluate(args)
    cmd_visualize(args)


def build_parser():
    p = argparse.ArgumentParser(description="QML on GSE84507 (CML) benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess", help="download data and build task .npz files").set_defaults(func=cmd_preprocess)
    sub.add_parser("list-models", help="list available algorithms").set_defaults(func=cmd_list_models)

    for cmd in ("train", "all"):
        sp = sub.add_parser(cmd, help="train models" if cmd == "train"
                            else "preprocess + train + evaluate + visualize")
        sp.add_argument("--model", nargs="+", default=["all"],
                        help="model name(s) or 'all' (see list-models)")
        sp.add_argument("--task", nargs="+", default=["all"],
                        help=f"task name(s) or 'all' {C.TASKS}")
        sp.add_argument("--qsvm-max-train", type=int, default=None,
                        help="cap on QSVM training cells per fold (default 250; 0=all)")
        sp.set_defaults(func=cmd_train if cmd == "train" else cmd_all)

    sub.add_parser("evaluate", help="print comparison table + markdown").set_defaults(func=cmd_evaluate)
    sub.add_parser("visualize", help="render all figures").set_defaults(func=cmd_visualize)
    return p


def main():
    args = build_parser().parse_args()
    # normalize qsvm-max-train: 0 means "use all"
    if getattr(args, "qsvm_max_train", None) == 0:
        args.qsvm_max_train = None
    args.func(args)


if __name__ == "__main__":
    main()
