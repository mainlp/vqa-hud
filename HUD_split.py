from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any
import argparse

@dataclass(frozen=True)
class QAItem:
    qid: str
    human_conf: Dict[str, float]
    raw: Dict[str, Any]

    @property
    def avg_conf(self) -> float:
        vals = list(self.human_conf.values())
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

def load_items(json_path: Path) -> List[QAItem]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[QAItem] = []
    for obj in data:
        # 容错：确保字段存在
        qid = str(obj.get("qid", ""))
        human_conf = obj.get("human_conf", {}) or {}
        if not isinstance(human_conf, dict):
            human_conf = {}
        items.append(QAItem(qid=qid, human_conf=human_conf, raw=obj))
    return items

def tercile_slices(n: int) -> Tuple[slice, slice, slice]:

    a = n // 3
    b = (n - a) // 2  # 剩余再平分
    c = n - a - b
    return slice(0, a), slice(a, a + b), slice(a + b, a + b + c)

def split_by_avg_conf(
    items: Iterable[QAItem],
    ascending: bool = True,
) -> Tuple[List[QAItem], List[QAItem], List[QAItem]]:

    sorted_items = sorted(items, key=lambda x: x.avg_conf, reverse=not ascending)
    s1, s2, s3 = tercile_slices(len(sorted_items))
    return (
        sorted_items[s1],  # low
        sorted_items[s2],  # mid
        sorted_items[s3],  # high
    )

def save_json(objs: List[QAItem], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([it.raw for it in objs], f, indent=2, ensure_ascii=False)

def summarize(bucket: List[QAItem], name: str) -> str:
    if not bucket:
        return f"{name}: 0 items"
    vals = [x.avg_conf for x in bucket]
    return (
        f"{name}: {len(bucket)} items | "
        f"avg={sum(vals)/len(vals):.4f}, "
        f"min={min(vals):.4f}, "
        f"max={max(vals):.4f}"
    )

def run(
    in_path: Path,
    low_out: Path,
    mid_out: Path,
    high_out: Path,
    ascending: bool = True,
) -> None:

    items = load_items(in_path)
    low, mid, high = split_by_avg_conf(items, ascending=ascending)

    save_json(low, low_out)
    save_json(mid, mid_out)
    save_json(high, high_out)

    all_vals = sorted([x.avg_conf for x in items])
    n = len(all_vals)
    if n:
        s1, s2, s3 = tercile_slices(n)
        thresholds = []
        if s1.stop - 1 >= 0:
            thresholds.append(("low_max", all_vals[s1.stop - 1]))
        if s2.stop - 1 >= 0:
            thresholds.append(("mid_max", all_vals[s2.stop - 1]))
        if s3.stop - 1 >= 0:
            thresholds.append(("high_max", all_vals[s3.stop - 1]))

        print(
            "Tercile thresholds (by avg_conf, "
            f"{'ascending' if ascending else 'descending'}):"
        )
        for k, v in thresholds:
            print(f"  {k}: {v:.6f}")

    print(summarize(low, "LOW"))
    print(summarize(mid, "MID"))
    print(summarize(high, "HIGH"))
    print(f"Saved:\n  {low_out}\n  {mid_out}\n  {high_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split VQA items into low/mid/high buckets by average human confidence."
    )
    p.add_argument("--in", dest="in_path", required=False, default="/user/project/vqa/beit_HUD.json")
    p.add_argument("--low_out", default="/nfs/gdata/user/project/vqa/conf24/models/unilm/beit3/eval_scores/beit_low.json")
    p.add_argument("--mid_out", default="/nfs/gdata/user/project/vqa/conf24/models/unilm/beit3/eval_scores/beit_mid.json")
    p.add_argument("--high_out", default="/nfs/gdata/user/project/vqa/conf24/models/unilm/beit3/eval_scores/beit_high.json")
    p.add_argument(
        "--ascending",
        action="store_true",
        help="Sort by avg_conf ascending (low→mid→high confidence). "
           ,
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    in_path = Path(args.in_path)
    low_out = Path(args.low_out)
    mid_out = Path(args.mid_out)
    high_out = Path(args.high_out)

    run(in_path, low_out, mid_out, high_out, ascending=args.ascending)
