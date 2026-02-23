import argparse
import csv
import datetime
import json
import logging
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from PIL import Image
from src.evaluation.lm_prediction_call import (
    LHRSLM,
    VHM,
    GeoChatLM,
    LMDeployLM,
    SkysenseGPTLM,
)
from src.evaluation.sc_models import LMDeployLMBatch
from src.tools.logger import setup_logger
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger("RSEval")


BENCH_DATASETS = {
    # cls
    "cls_aid": ("cls_AID.json", "cls"),
    "cls_METER_ML": ("cls_METER_ML.json", "cls"),
    "cls_NWPU_RESISC45": ("cls_NWPU_RESISC45.json", "cls"),
    "cls_SIRI": ("cls_SIRI_WHU.json", "cls"),
    "cls_WHU_RS19": ("cls_WHU_RS19.json", "cls"),
    # rsvqa
    "vqa_HR-comp": ("RSVQA_HR-comp_RSVQA.json", "vqa"),
    "vqa_HR-pre": ("RSVQA_HR-presence_RSVQA.json", "vqa"),
    "vqa_LR-comp": ("RSVQA_LR-comp_RSVQA.json", "vqa"),
    "vqa_LR-pre": ("RSVQA_LR-presence_RSVQA.json", "vqa"),
    "vqa_LR-rural": ("RSVQA_LR-rural_urban_RSVQA.json", "vqa"),
    # custom vqa
    "VQA_self-eval": ("VQA_self-eval.json", "vqa"),
    # custom cls
    "CLS_self-eval": ("CLS_self-eval.json", "cls"),
    # custom vg
    "VG_self-Eval": ("VG_self-Eval.json", "bbox"),
    # vg
    "rs_vg": ("VG_DOIR_RSVG_test.json", "bbox"),
    # LHRS-Bench
    "lhrs_bench": ("LHRS-Bench.json", "lhrsbench"),
}

MODEL_TYPE_MAP = {
    "lhrs": LHRSLM,
    "vhm": VHM,
    "skysensegpt": SkysenseGPTLM,
    "geochat": GeoChatLM,
    "lmdeploy": LMDeployLMBatch,           # batched: all N samples in one pipeline call
    "lmdeploy_reasoning": LMDeployLMBatch,  # batched: all N samples in one pipeline call
}

LHRS_TYPE_MAP = {
    "1": "identity",
    "2": "color",
    "3": "orientation",
    "4": "shape",
    "5": "quantity",
    "6": "area",
    "7": "distance",
    "8": "resolution",
    "9": "modality",
    "10": "location",
    "11": "reasoning",
}


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--model_type",
        type=str,
        default="lhrs",
        choices=[
            "lhrs",
            "vhm",
            "skysensegpt",
            "geochat",
            "lmdeploy",
        ],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--force_inference", type=bool, default=False)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit the number of data to evaluate",
    )
    parser.add_argument("--reasoning_config", type=str, default=None)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=7,
        help="number of samples to generate per image for self-consistency majority voting",
    )
    return parser.parse_args()


def parse_sc_answer(prediction):
    """Parse a single raw prediction into a cleaned answer string for voting."""
    if "<reasoning>" in prediction:
        try:
            prediction = prediction.split("<reasoning>")[1].split("</reasoning>")[1]
        except Exception:
            pass

    if "<answer>" in prediction:
        try:
            prediction = prediction.split("<answer>")[1].split("</answer>")[0]
        except Exception:
            pass

    prediction = prediction.replace(" ", "")

    while " " in prediction:
        prediction = prediction.replace(" ", "")

    if "." in prediction:
        prediction = prediction.split(".")[0]

    if "," in prediction:
        prediction = prediction.split(",")[0]

    return prediction.strip().lower()


def majority_vote(predictions):
    """
    Given a list of N raw prediction strings, parse each one and return the raw
    prediction whose cleaned answer wins the majority vote.

    Returns:
        (winning_raw_pred, winning_cleaned_answer, vote_counts_dict)
    """
    cleaned = [parse_sc_answer(p) for p in predictions]
    vote_counts = Counter(cleaned)
    winner_clean = vote_counts.most_common(1)[0][0]

    # Return the first raw prediction whose cleaned answer matches the winner
    winning_raw = predictions[0]
    for raw, clean in zip(predictions, cleaned):
        if clean == winner_clean:
            winning_raw = raw
            break

    return winning_raw, winner_clean, dict(vote_counts)


def eval_results_vqa_sc(args, result_json_file, save_metric_json):
    with open(result_json_file, "r") as f:
        result_lines = f.readlines()

    ret = {}
    final_dict = defaultdict(list)

    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        answer = str(result_dict["answer"])

        # Use pre-voted `preds` list if available, else fall back to single `pred`
        if "preds" in result_dict and isinstance(result_dict["preds"], list):
            preds_list = result_dict["preds"]
        else:
            preds_list = [str(result_dict["pred"])]

        _, voted_clean, vote_counts = majority_vote(preds_list)
        prediction = voted_clean

        answer_clean = answer.replace(" ", "")
        while " " in answer_clean:
            answer_clean = answer_clean.replace(" ", "")
        answer_clean = answer_clean.strip().lower()

        final_dict["score"].append(prediction in answer_clean)
        final_dict["filename"].append(str(os.path.basename(result_dict["filename"])))
        final_dict["size"].append(str(result_dict["size"]))
        final_dict["query"].append(str(result_dict["query"]))
        final_dict["answer"].append(str(result_dict["answer"]))
        final_dict["prediction"].append(str(prediction))
        final_dict["raw_prediction"].append(str(result_dict.get("pred", preds_list[0])))
        final_dict["votes"].append(str(vote_counts))
        final_dict["n_samples"].append(len(preds_list))

    avg_score = sum(final_dict["score"]) / len(final_dict["score"])
    perf_dict = {
        "accuracy": avg_score,
    }
    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    ret.to_json(save_metric_json, orient="records", indent=4)
    logger.info(f"accuracy (self-consistency n={args.n_samples}): {avg_score}")
    return perf_dict


def eval_results_lhrsbench_sc(args, result_json_file, save_metric_json):
    with open(result_json_file, "r") as f:
        result_lines = f.readlines()

    ret = {}
    final_dict = defaultdict(list)
    type_level_score = defaultdict(list)

    for img_i, line in enumerate(result_lines):
        result_dict = json.loads(line)
        answer = str(result_dict["answer"])

        if "preds" in result_dict and isinstance(result_dict["preds"], list):
            preds_list = result_dict["preds"]
        else:
            preds_list = [str(result_dict["pred"])]

        _, voted_clean, vote_counts = majority_vote(preds_list)
        prediction = voted_clean

        answer_clean = answer.replace(" ", "").replace(".", "").lower()

        types = result_dict["type"]
        for t in types:
            type_level_score[t].append(prediction in answer_clean)

        final_dict["score"].append(prediction in answer_clean)
        final_dict["filename"].append(str(os.path.basename(result_dict["filename"])))
        final_dict["size"].append(str(result_dict["size"]))
        final_dict["query"].append(str(result_dict["query"]))
        final_dict["answer"].append(str(result_dict["answer"]))
        final_dict["prediction"].append(str(prediction))
        final_dict["raw_prediction"].append(str(result_dict.get("pred", preds_list[0])))
        final_dict["type"].append(result_dict["type"])
        final_dict["votes"].append(str(vote_counts))

    for t in type_level_score:
        type_level_score[t] = sum(type_level_score[t]) / len(type_level_score[t])

    avg_score = sum(final_dict["score"]) / len(final_dict["score"])
    perf_dict = {"accuracy": avg_score}
    for t in type_level_score:
        perf_dict[f"accuracy_{LHRS_TYPE_MAP[t]}"] = type_level_score[t]

    ret.update(perf_dict)
    ret.update(final_dict)

    ret = pd.DataFrame({x: ret[x] for x in ret})
    ret.to_json(save_metric_json, orient="records", indent=4)

    logger.info(f"accuracy (self-consistency n={args.n_samples}): {avg_score}")
    for t in type_level_score:
        logger.info(f"accuracy_{LHRS_TYPE_MAP[t]}: {type_level_score[t]}")
    return perf_dict


def convt_qa(conversations, task_type, model):
    values = [conversation["value"] for conversation in conversations]
    query = values[0]
    answer = values[1]

    classification_prefix = getattr(model, "cls_prefix", "")
    vqa_prefix = getattr(model, "vqa_prefix", "")
    vqa_suffix = getattr(model, "vqa_suffix", "")
    vg_prefix = getattr(model, "vg_prefix", "")
    vg_suffix = getattr(model, "vg_suffix", "")
    if "cls" in task_type:
        query = classification_prefix + " " + query
    elif "vqa" in task_type:
        query = vqa_prefix + " " + query + " " + vqa_suffix
    elif "bbox" in task_type:
        query = vg_prefix + " " + query + " " + vg_suffix

    return query, answer


def infer_single_sc(args, model, anns_json_path, anns, task_type):
    fn = anns["image"]
    if "image_path" not in anns.keys():
        json_pathlib_obj = Path(anns_json_path)
        fn_full = json_pathlib_obj.parent / json_pathlib_obj.stem / fn
    elif Path(anns["image_path"]).is_absolute():
        fn_full = Path(anns["image_path"]) / fn
    else:
        dataset_base = Path(anns_json_path).parent
        fn_full = dataset_base / anns["image_path"] / fn

    question, answer = convt_qa(anns["conversations"], task_type, model)
    question = question.replace("<image>\n", "")
    if "size" not in anns.keys():
        image = Image.open(fn_full).convert("RGB")
        anns["size"] = image.size

    result_dict = {
        "filename": str(fn_full),
        "size": anns["size"],
        "query": question,
        "answer": answer,
    }

    if "type" in anns.keys():
        result_dict["type"] = anns["type"]

    try:
        outputs = model.generate_n(
            prompt=question, image_files=fn_full, n_samples=args.n_samples
        )
        # outputs is a list of n_samples raw strings
        if not isinstance(outputs, list):
            outputs = [outputs]

        winning_raw, winning_clean, vote_counts = majority_vote(outputs)

        result_dict["preds"] = outputs          # all N raw outputs
        result_dict["pred"] = winning_raw       # majority-voted raw output
        result_dict["voted_answer"] = winning_clean
        result_dict["vote_counts"] = vote_counts
    except Exception as e:
        logger.error(f"Error processing {fn_full}: {str(e)}")
        result_dict["preds"] = [f"ERROR: {str(e)}"]
        result_dict["pred"] = f"ERROR: {str(e)}"
        result_dict["error"] = True

    return result_dict


def infer_model_sc(
    args, model, anns_json_path, task_type, save_json, local_rank, world_size
):
    with open(anns_json_path, "r") as f:
        anns_dict = json.load(f)

    if args.limit is not None:
        anns_dict = anns_dict[: args.limit]

    chunk_size = len(anns_dict) // world_size
    sub_lists = [
        anns_dict[i : i + chunk_size] for i in range(0, len(anns_dict), chunk_size)
    ]
    if len(anns_dict) % world_size != 0:
        sub_lists[-2] = sub_lists[-2] + sub_lists[-1]

    sub_anns_dict = sub_lists[local_rank]

    final_results = []
    for idx, anns in tqdm(enumerate(sub_anns_dict), total=len(sub_anns_dict)):
        result_dict = infer_single_sc(args, model, anns_json_path, anns, task_type)

        final_results.append(json.dumps(result_dict))

        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    if world_size > 1:
        model.accelerator.wait_for_everyone()
        if local_rank == 0:
            gathered_objects = (
                model.accelerator.gather(final_results).cpu().detach().numpy().tolist()
            )

            final_results = []
            for sublist in gathered_objects:
                final_results.extend(sublist)

    if local_rank == 0:
        with open(save_json, "w") as f:
            f.write("\n".join(final_results))


def eval_task_sc(args, model, task_index, local_rank, world_size):
    anns_json, task_type = BENCH_DATASETS[task_index]

    anns_json_path = Path(args.data_root) / anns_json

    test_name = Path(anns_json_path).stem

    if args.model_type == "lmdeploy":
        model_name = Path(args.model_path).stem
    else:
        model_name = args.model_type

    # SC-specific output filenames include n_samples count
    save_json = (
        Path(args.output_dir) / f"{test_name}_{model_name}_sc{args.n_samples}_eval.jsonl"
    )
    save_metric_json = (
        Path(args.output_dir) / f"{test_name}_{model_name}_sc{args.n_samples}_eval.json"
    )

    if not save_json.exists() or args.force_inference:
        infer_model_sc(
            args,
            model,
            anns_json_path,
            task_type,
            save_json,
            local_rank,
            world_size,
        )
    else:
        with open(anns_json_path, "r") as f:
            anns_dict = json.load(f)
        with open(save_json, "r") as f:
            save_dict = f.readlines()
        if len(anns_dict) != len(save_dict):
            infer_model_sc(
                args,
                model,
                anns_json_path,
                task_type,
                save_json,
                local_rank,
                world_size,
            )

    if local_rank == 0:
        if task_type in ("cls", "vqa"):
            return eval_results_vqa_sc(args, save_json, save_metric_json)
        elif task_type == "lhrsbench":
            return eval_results_lhrsbench_sc(args, save_json, save_metric_json)
    return None


if __name__ == "__main__":
    args = arg_parser()
    output_dir = Path(args.output_dir)

    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    setup_logger("RSEval", args.output_dir, rank=accelerator.local_process_index)
    logger.info(f"Args: {args}")
    logger.info(
        f"Self-Consistency: n_samples={args.n_samples}, temperature={args.temperature}"
    )

    device = accelerator.device
    extra_kwargs = {}
    if args.reasoning_config is not None:
        extra_kwargs["reasoning_config"] = args.reasoning_config
    model = MODEL_TYPE_MAP[args.model_type](
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        beam_size=args.beam_size,
        do_sample=args.do_sample,
        use_cache=args.use_cache,
        dtype=args.dtype,
        device=device,
        max_new_tokens=args.max_new_tokens,
        **extra_kwargs,
    )

    model.accelerator = accelerator
    model.rank = accelerator.local_process_index
    model.world_size = accelerator.num_processes

    total_metrics = {}
    if args.task == "all":
        for task_index in BENCH_DATASETS.keys():
            if accelerator.is_main_process:
                logger.info(f"===================={task_index}======================")
                metrics = eval_task_sc(
                    args,
                    model,
                    task_index,
                    accelerator.local_process_index,
                    accelerator.num_processes,
                )
                total_metrics.update({task_index: metrics})
    else:
        metrics = eval_task_sc(
            args,
            model,
            args.task,
            accelerator.local_process_index,
            accelerator.num_processes,
        )
        total_metrics.update({args.task: metrics})
