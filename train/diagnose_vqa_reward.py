#!/usr/bin/env python
"""Why does the VQA GRPO reward come out at exactly 0.0 for most prompts?

The eval dump for the same checkpoint (qwen35-cotall-vqa-merged) shows 950/950 completions
matching <reasoning>...</reasoning><answer>...</answer> under an anchored re.match, so the
model plainly can produce the format. Yet scoring the GRPO corpus with grpo.py's reward
functions gave 340/363 prompts a mean reward of 0.0 -- not even the 0.2 format component.

Something between the two setups differs. The candidates, and what this script varies:
  - system turn : GRPO's prompt column carries grpo.py's SYSTEM_PROMPT; the TinyRS eval
                  sends none at all
  - decoding    : GRPO samples at temperature 1.0; the eval is greedy
Each cell prints the raw completion and the reward components, so the cause is visible
rather than inferred.
"""
import argparse
import glob
import io
import json
import re
from pathlib import Path

import pyarrow as pa
from PIL import Image

FORMAT_RE = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"


def read_split(dataset_dir):
    split_dir = Path(dataset_dir) / "train"
    state = json.loads((split_dir / "state.json").read_text())
    tabs = []
    for e in state["_data_files"]:
        with pa.memory_map(str(split_dir / e["filename"]), "rb") as src:
            tabs.append(pa.ipc.open_stream(src).read_all())
    t = pa.concat_tables(tabs)
    idx = state.get("_indices_data_files") or []
    if idx:
        its = []
        for e in idx:
            with pa.memory_map(str(split_dir / e["filename"]), "rb") as src:
                its.append(pa.ipc.open_stream(src).read_all())
        t = t.take(pa.concat_tables(its).column(0).to_pylist())
    return t.to_pylist()


def acc_reward(content, sol):
    m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    student = m.group(1).strip() if m else content.strip()
    if sol[0] == "1":
        return 2.0 if student.lower().replace('.', '') == sol[1].strip().lower().replace('.', '') else 0.0
    return 0.0


def fmt_reward(content):
    return 0.2 if re.match(FORMAT_RE, content, re.DOTALL) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_prompts", type=int, default=6)
    ap.add_argument("--max_pixels", type=int, default=1003520)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    rows = read_split(args.dataset)[: args.n_prompts]
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                         max_pixels=args.max_pixels)
    llm = LLM(model=args.model, trust_remote_code=True, gpu_memory_utilization=0.90,
              max_model_len=8192, limit_mm_per_prompt={"image": 1})

    def build(row, with_system):
        msgs = []
        for m in row["prompt"]:
            if m["role"] == "system" and not with_system:
                continue
            content = []
            for c in m["content"]:
                if c.get("text") is not None:
                    content.append({"type": "text", "text": c["text"]})
                else:
                    content.append({"type": "image"})
            msgs.append({"role": m["role"], "content": content})
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        return {"prompt": text, "multi_modal_data": {"image": img}}

    for with_system in (True, False):
        for temp in (0.0, 1.0):
            tag = f"system={'YES' if with_system else 'NO ' }  temp={temp}"
            sp = SamplingParams(n=1, temperature=temp, max_tokens=1024)
            outs = llm.generate([build(r, with_system) for r in rows], sampling_params=sp)
            fmt_ok = acc_ok = 0
            print(f"\n{'='*88}\n### {tag}\n{'='*88}")
            for r, o in zip(rows, outs):
                txt = o.outputs[0].text
                f, a = fmt_reward(txt), acc_reward(txt, r["solution"])
                fmt_ok += f > 0
                acc_ok += a > 0
                print(f"\n  gold={r['solution'][1]!r}  format={f}  accuracy={a}  total={f+a}"
                      f"  chars={len(txt)}  finish={o.outputs[0].finish_reason}"
                      f"  has_close_reasoning={'</reasoning>' in txt}  has_answer={'<answer>' in txt}")
                print(f"  head={txt[:150]!r}")
                print(f"  TAIL={txt[-260:]!r}")
            print(f"\n  --> format ok {fmt_ok}/{len(rows)}   accuracy ok {acc_ok}/{len(rows)}")


if __name__ == "__main__":
    main()
