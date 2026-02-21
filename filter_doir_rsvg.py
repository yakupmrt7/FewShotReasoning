import json
import re
import os

INPUT_PATH = "/arf/scratch/aalatan/VHM_dataset_sft_reasoning/list_sft_reasoning_big_filtered.json"
OUTPUT_DIR = "/arf/scratch/aalatan/Datasets_Self-Eval_VG"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "DOIR-RSVG_filtered.json")

SCALE = 800 / 1000  # 0.8


def scale_coords(match):
    """Scale [[x1,y1,x2,y2]] coordinates from 0-1000 to 0-800."""
    coords = [int(match.group(i)) for i in range(1, 5)]
    scaled = [round(c * SCALE) for c in coords]
    return f"[[{scaled[0]},{scaled[1]},{scaled[2]},{scaled[3]}]]"


def process_text(text):
    """Replace all [[x,y,x,y]] coordinate patterns in a text string."""
    return re.sub(r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]', scale_coords, text)


def main():
    print(f"Loading {INPUT_PATH} ...")
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    filtered = [s for s in data if s.get("image", "").startswith("DOIR-RSVG")]
    print(f"DOIR-RSVG samples: {len(filtered)}")

    for sample in filtered:
        for turn in sample.get("conversations", []):
            if "value" in turn:
                turn["value"] = process_text(turn["value"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
