#!/usr/bin/env python
"""Find the GRPO prompts that produce no learning signal (stage 1 of 2).

GRPO's advantage is (r_i - mean(r)) / std(r) computed *within* each prompt's group of
`num_generations` completions. When every completion in a group earns the same reward the
std is 0, all advantages are 0, and the prompt contributes nothing to the gradient. TRL
logs how often that happens as `frac_reward_zero_std`, and on these datasets it is severe
and gets worse as training proceeds:

    VQA  0.91 -> 0.98      2-9%  of prompts useful
    CLS  0.72 -> 0.78     22-28% of prompts useful
    VG   0.08 -> 0.13     87-92% of prompts useful

which is the same ordering as which tasks GRPO actually moved. This samples the pre-GRPO
policy exactly as training will, scores the completions with grpo.py's own reward
functions, and records which prompts have non-zero group spread.

Runs under recot-eval (vLLM), the same env the TinyRS evals use. That env cannot
`import datasets` -- it ships datasets 1.1.1 against pyarrow 24, where `pa.PyExtensionType`
no longer exists -- so the corpus is read straight off the arrow file with pyarrow and this
stage only writes an index list. materialize_filtered_dataset.py turns that into a real
dataset from an env with a working `datasets`.

The reward code is copied from src/open_r1/grpo.py rather than imported, because importing
it pulls in trl/deepspeed. If the reward functions there change, change them here too.
"""
import argparse
import glob
import io
import json
import re
import statistics as stats
from pathlib import Path

import pyarrow as pa
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

FORMAT_RE = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"


# ---------------------------------------------------------------- rewards
# verbatim from grpo.py: accuracy_reward (x2.0 weight) + format_reward (0.2)

def accuracy_reward_one(content, sol):
    reward = 0.0
    answer_type, answer = sol[0], sol[1]

    m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    student_answer = m.group(1).strip() if m else content.strip()

    if answer_type == "1":                      # exact match
        if student_answer.lower().replace('.', '') == answer.strip().lower().replace('.', ''):
            reward = 1.0
    elif answer_type == "2":                    # bounding box -> IoU
        try:
            nums = [int(n) for n in re.findall(r'\d+', student_answer)]
            sb = [nums[i:i+4] for i in range(0, len(nums), 4)][0]
            gnums = [int(n) for n in re.findall(r'\d+', answer)]
            gb = [gnums[i:i+4] for i in range(0, len(gnums), 4)][0]
            x1, y1 = max(sb[0], gb[0]), max(sb[1], gb[1])
            x2, y2 = min(sb[2], gb[2]), min(sb[3], gb[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = ((sb[2]-sb[0])*(sb[3]-sb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter)
            reward = inter / union if union > 0 else 0.0
        except Exception:
            reward = 0.0
    return reward * 2.0


def format_reward_one(content):
    return 0.2 if re.match(FORMAT_RE, content, re.DOTALL) else 0.0


def total_reward(content, sol):
    return accuracy_reward_one(content, sol) + format_reward_one(content)


# ---------------------------------------------------------------- io
def read_split(dataset_dir):
    """Rows straight off the arrow shards, bypassing the `datasets` library.

    Shards come from state.json, never from a glob: these directories can also contain
    `cache-*.arrow` files left behind by earlier map/filter calls, which have a different
    schema and different row counts. Globbing picked up a 1003-row cache next to the 843
    real rows in the VG set and produced a KeyError on the first missing column.

    An `_indices_data_files` entry means the split is a selection over the shards, so the
    rows have to be gathered through that mapping to match what `datasets` reports.
    """
    split_dir = Path(dataset_dir) / "train"
    state_path = split_dir / "state.json"
    if not state_path.exists():
        raise SystemExit(f"no state.json under {split_dir}")
    state = json.loads(state_path.read_text())

    def read_files(entries):
        out = []
        for e in entries:
            with pa.memory_map(str(split_dir / e["filename"]), "rb") as src:
                out.append(pa.ipc.open_stream(src).read_all())
        return pa.concat_tables(out) if out else None

    table = read_files(state.get("_data_files", []))
    if table is None:
        raise SystemExit(f"state.json lists no data files for {split_dir}")

    idx_files = state.get("_indices_data_files") or []
    if idx_files:
        idx_table = read_files(idx_files)
        indices = idx_table.column(0).to_pylist()
        table = table.take(indices)

    return table.to_pylist()


def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, help="the pre-GRPO policy this dataset will train")
    p.add_argument("--keep_json", required=True, help="where to write the kept indices")
    p.add_argument("--num_generations", type=int, default=8,
                   help="must match the GRPO run's --num_generations, or the measured group "
                        "spread is not the spread training would see")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="must match the GRPO run's --temperature, for the same reason")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=1024,
                   help="training allows 8192, but logged completions run ~250-750 tokens")
    p.add_argument("--max_pixels", type=int, default=1003520,
                   help="match the GRPO run's --max_pixels so images tokenize identically")
    p.add_argument("--min_std", type=float, default=1e-6)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = arg_parser()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    rows = read_split(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[filter] {args.dataset}: {len(rows)} prompts | model={args.model}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                              max_pixels=args.max_pixels)
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.90,
              max_model_len=8192, limit_mm_per_prompt={"image": 1})
    sampling = SamplingParams(n=args.num_generations, temperature=args.temperature,
                              top_p=args.top_p, max_tokens=args.max_new_tokens)

    # the `prompt` column already carries the system turn grpo.py prepends, so rendering it
    # as-is reproduces training's input exactly
    inputs = []
    for r in rows:
        msgs = []
        for m in r["prompt"]:
            content = []
            for c in m["content"]:
                if c.get("text") is not None:
                    content.append({"type": "text", "text": c["text"]})
                else:
                    content.append({"type": "image"})
            msgs.append({"role": m["role"], "content": content})
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        inputs.append({"prompt": text, "multi_modal_data": {"image": img}})

    outs = llm.generate(inputs, sampling_params=sampling)

    keep, per_prompt, clipped = [], [], 0
    for i, (r, out) in enumerate(zip(rows, outs)):
        sol = r["solution"]
        rewards = []
        for o in out.outputs:
            rewards.append(total_reward(o.text, sol))
            if o.finish_reason == "length":
                clipped += 1
        sd = stats.pstdev(rewards)
        per_prompt.append({"idx": i, "mean": sum(rewards)/len(rewards), "std": sd,
                           "min": min(rewards), "max": max(rewards)})
        if sd > args.min_std:
            keep.append(i)

    n = len(rows)
    n_zero = n - len(keep)
    always = sum(1 for p in per_prompt if p["std"] <= args.min_std and p["mean"] > 1.0)
    never = n_zero - always

    print(f"\n[filter] group reward spread over {n} prompts")
    print(f"    zero-variance (drop) : {n_zero:5d}  ({100*n_zero/n:5.1f}%)")
    print(f"        always solved    : {always:5d}   nothing left to learn")
    print(f"        never solved     : {never:5d}   no gradient direction")
    print(f"    kept (useful)        : {len(keep):5d}  ({100*len(keep)/n:5.1f}%)")
    print(f"    completions clipped at {args.max_new_tokens} tok: {clipped}/{n*args.num_generations}")
    print(f"    frac_reward_zero_std: {n_zero/n:.3f} now -> ~0.0 on the filtered set", flush=True)

    if not keep:
        print("[filter] WARNING: no prompt has any reward spread. A smaller dataset cannot "
              "fix this -- the reward itself is saturated and needs to be made denser.",
              flush=True)

    Path(args.keep_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.keep_json, "w") as f:
        json.dump({"dataset": args.dataset, "model": args.model, "n": n,
                   "num_generations": args.num_generations, "temperature": args.temperature,
                   "kept": keep, "dropped_zero_std": n_zero,
                   "always_solved": always, "never_solved": never,
                   "per_prompt": per_prompt}, f, indent=2)
    print(f"[filter] wrote {args.keep_json}", flush=True)


if __name__ == "__main__":
    main()
