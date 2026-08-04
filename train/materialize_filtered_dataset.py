#!/usr/bin/env python
"""Turn filter_grpo_by_difficulty.py's index list into a real dataset (stage 2 of 2).

Split out from stage 1 purely because of an env constraint: the vLLM env (recot-eval)
cannot `import datasets`, so it can only emit indices. This runs anywhere that has a
working `datasets` -- recot-train-grpo, the same env GRPO training uses -- and needs no GPU.
"""
import argparse
import json

from datasets import DatasetDict, load_from_disk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_json", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    meta = json.load(open(args.keep_json))
    keep = meta["kept"]
    if not keep:
        raise SystemExit("[materialize] index list is empty -- refusing to write an empty "
                         "dataset. The reward is saturated; make it denser instead.")

    ds = load_from_disk(meta["dataset"])
    split = ds["train"]
    if len(split) != meta["n"]:
        raise SystemExit(f"[materialize] {meta['dataset']} has {len(split)} rows but the "
                         f"filter saw {meta['n']} -- the dataset changed since filtering")

    # Save as a DatasetDict, not a bare Dataset: grpo.py does
    # dataset[script_args.dataset_train_split], which raises
    # "Column 'train' doesn't exist" on a plain Dataset. Every other split in the source
    # is carried over untouched so the filtered copy is a drop-in replacement.
    out = split.select(keep)
    dd = DatasetDict({k: (out if k == "train" else v) for k, v in ds.items()})
    dd.save_to_disk(args.out)
    print(f"[materialize] {meta['n']} -> {len(out)} prompts "
          f"({100*len(out)/meta['n']:.1f}% kept)  -> {args.out}  splits={list(dd.keys())}")
    print(f"[materialize]   dropped {meta['dropped_zero_std']} zero-variance "
          f"({meta['always_solved']} always solved, {meta['never_solved']} never solved)")


if __name__ == "__main__":
    main()
