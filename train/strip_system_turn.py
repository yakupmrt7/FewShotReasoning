#!/usr/bin/env python
"""Remove the system turn from a GRPO dataset's `prompt` column.

grpo.py's dataset builder prepends its SYSTEM_PROMPT to every prompt. For the CLS and VG
cotall checkpoints that is harmless -- they still emit
<reasoning>...</reasoning><answer>...</answer> and earn reward. The VQA cotall checkpoint
does not: with the system turn present it opens <reasoning>, writes the reasoning, then
jumps straight to </answer>, never closing </reasoning>, never opening <answer>, and never
writing the yes/no. Both reward components therefore score 0, on 340 of 363 prompts.

Measured on qwen35-cotall-vqa-merged, 3 prompts x {greedy, T=1.0}:

    system turn present : format 0/6   accuracy 0/6
    system turn removed : format 6/6   accuracy 5/6

The cause is distribution mismatch: the model's SFT and CoT stages were trained with no
system turn at all (Qwen3.5 gets no default system message), and the TinyRS eval sends
none either. The system turn is an unseen prefix, and this checkpoint responds to it by
corrupting its output structure. Stripping it makes GRPO's inputs match both the model's
training distribution and the eval, which is the setting the reward function was written for.
"""
import argparse

from datasets import DatasetDict, load_from_disk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_proc", type=int, default=4)
    args = p.parse_args()

    ds = load_from_disk(args.dataset)
    split = ds["train"]

    before = sum(1 for m in split[0]["prompt"] if m["role"] == "system")
    print(f"[strip] {args.dataset}: {len(split)} rows, "
          f"{before} system turn(s) in the first prompt")

    def drop_system(row):
        row["prompt"] = [m for m in row["prompt"] if m["role"] != "system"]
        return row

    out = split.map(drop_system, num_proc=args.num_proc,
                    desc="dropping system turn")

    roles = [m["role"] for m in out[0]["prompt"]]
    if "system" in roles:
        raise SystemExit("[strip] system turn survived the map -- aborting")
    print(f"[strip] remaining roles in first prompt: {roles}")

    dd = DatasetDict({k: (out if k == "train" else v) for k, v in ds.items()})
    dd.save_to_disk(args.out)
    print(f"[strip] wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
