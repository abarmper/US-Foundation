"""Unified CLI:  python -m gubiometry <cmd> --config cfg.yaml [-o key=val ...]

Commands: phase1 | phase2 | kfold | predict | evaluate | make-splits
"""

import argparse

from .config import load_config, apply_overrides, RunConfig


def _load(args):
    cfg = load_config(args.config) if args.config else RunConfig()
    apply_overrides(cfg, args.override)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(prog="gubiometry", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("phase1", "phase2", "kfold", "predict", "evaluate"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=None, help="YAML config path")
        sp.add_argument("--override", "-o", action="append", default=[],
                        help="dotted override, e.g. -o optim.epochs=1 -o model.backbone.name=dummy")

    ms = sub.add_parser("make-splits")
    ms.add_argument("--data-root", default="./data")
    ms.add_argument("--kfold", action="store_true", help="write stratified k-fold splits")
    ms.add_argument("--n-splits", type=int, default=5)
    ms.add_argument("--seed", type=int, default=42)

    args = p.parse_args(argv)

    if args.cmd == "make-splits":
        from .data.splits import make_holdout_split, make_kfold_splits
        if args.kfold:
            paths = make_kfold_splits(args.data_root, args.n_splits, args.seed)
            print(f"Wrote {len(paths)} fold files under {args.data_root}/splits/kfold_v1/")
        else:
            path, counts = make_holdout_split(args.data_root, seed=args.seed)
            print(f"Wrote {path}: {counts}")
        return

    cfg = _load(args)
    if args.cmd == "phase1":
        from .engine import phase1
        phase1.train(cfg)
    elif args.cmd == "phase2":
        from .engine import phase2
        phase2.train(cfg)
    elif args.cmd == "kfold":
        from .engine import kfold
        kfold.run(cfg)
    elif args.cmd == "predict":
        from .engine import predict
        predict.run(cfg)
    elif args.cmd == "evaluate":
        from .engine import evaluate
        evaluate.run(cfg)


if __name__ == "__main__":
    main()
