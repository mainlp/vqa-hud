from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple
import numpy as np
from scipy.special import softmax


def load_beit_label_dict(p: Path) -> Dict[str, int]:
    with p.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    d: Dict[str, int] = {}
    for line in lines:
        obj = json.loads(line.strip("\n"))
        d[str(obj["answer"])] = int(obj["label"])
    return d


def load_lxmert_label_dict(p: Path) -> Dict[str, int]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def load_predictions(p: Path) -> List[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_annotations(p: Path) -> Dict[int, List[Dict[str, Any]]]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(a["question_id"]): a["answers"] for a in data["annotations"]}


def human_scores(ans_list: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    acc: Dict[str, Tuple[float, int]] = {}
    w = {"yes": 1.0, "maybe": 0.5, "no": 0.01}
    for a in ans_list:
        k = str(a["answer"])
        s, n = acc.get(k, (0.0, 0))
        acc[k] = (s + w.get(str(a["answer_confidence"]), 0.0), n + 1)
    return {k: s / n for k, (s, n) in acc.items() if n > 0}


def top_prob(logits: Iterable[float]) -> float:
    v = np.asarray(list(logits), dtype=np.float64)
    return float(softmax(v, axis=0).max())


def run(
    beit_labelpath: Path,
    lxmert_labelpath: Path,
    pre_file: Path,
    src_file: Path,
    output_path: Path,
    label_source: str = "lxmert",
) -> None:
    beit_labels = load_beit_label_dict(beit_labelpath)
    lxmert_labels = load_lxmert_label_dict(lxmert_labelpath)
    label_dict = lxmert_labels if label_source.lower() == "lxmert" else beit_labels

    preds = load_predictions(pre_file)
    ann = load_annotations(src_file)

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(preds, 1):
        qid = int(item["question_id"])
        ans = item.get("answer")
        logits = item["all"]
        prob = top_prob(logits)
        hum = human_scores(ann[qid])
        count = sum(1 for k in hum.keys() if k in label_dict)
        if count > 1:
            out.append(
                {
                    "qid": qid,
                    "ans": ans,
                    "model_conf": prob,
                    "human_conf": hum,
                    "model_all": logits,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--beit_labelpath", default="/data/vqa2.0/answer2label.txt")
    p.add_argument("--lxmert_labelpath", default="/data/vqa/trainval_ans2label.json")
    p.add_argument("--pre_file", default="/beit3/prediction/base/submit_vqav2_val.json")
    p.add_argument("--src_file", default="/v2_mscoco_val2014_annotations.json")
    p.add_argument("--output_path", default="/beit_HUD.json")
    p.add_argument("--label_source", choices=["lxmert", "beit"], default="lxmert")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        Path(args.beit_labelpath),
        Path(args.lxmert_labelpath),
        Path(args.pre_file),
        Path(args.src_file),
        Path(args.output_path),
        label_source=args.label_source,
    )
