from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy as kl_div

EPS = 1e-12


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


def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def human_annotations_map(annotations_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    data = load_json(annotations_path)
    return {int(a["question_id"]): a["answers"] for a in data["annotations"]}


def softmax_probs(logits: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    v = np.asarray(list(logits), dtype=np.float64)
    v = v / max(temperature, EPS)
    return softmax(v, axis=0)


def normalize(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return x / s


def vqa2_accuracy(human_answers: List[Dict[str, Any]], pred: str) -> float:
    matches = sum(1 for a in human_answers if str(a["answer"]) == str(pred))
    return min(matches / 3.0, 1.0)


def ece_score(
    items: List[Dict[str, Any]],
    n_bins: int,
    label_dict: Dict[str, int],
    annotations_path: Path,
    use_ts: bool = False,
    temperature: float = 1.0,
) -> Tuple[float, float, float, List[float], List[float], List[int], int, float]:
    human_map = human_annotations_map(annotations_path)
    confs: List[float] = []
    accs: List[float] = []

    for it in items:
        logits = it["model_all"]
        probs = softmax_probs(logits, temperature if use_ts else 1.0)
        qid = int(it["qid"])
        ans = str(it["ans"])
        idx = label_dict.get(ans, None)
        if idx is None or idx >= len(probs):
            continue
        confs.append(float(probs[idx]))
        accs.append(vqa2_accuracy(human_map[qid], ans))

    confidences = np.asarray(confs, dtype=np.float64)
    scores = np.asarray(accs, dtype=np.float64)
    if len(confidences) == 0:
        return 0.0, 0.0, 0.0, [], [], [0] * n_bins, n_bins, temperature

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(confidences, bin_edges, right=True) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    final = 0.0
    rt_acc: List[float] = []
    rt_conf: List[float] = []
    rt_count: List[int] = []
    for b in range(n_bins):
        mask = bins == b
        cnt = int(np.sum(mask))
        rt_count.append(cnt)
        if cnt == 0:
            rt_acc.append(0.0)
            rt_conf.append(0.0)
            continue
        acc_b = float(np.mean(scores[mask]))
        conf_b = float(np.mean(confidences[mask]))
        prop = cnt / len(confidences)
        final += abs(conf_b - acc_b) * prop
        rt_acc.append(acc_b)
        rt_conf.append(conf_b)

    return final, float(confidences.min()), float(confidences.max()), rt_acc, rt_conf, rt_count, n_bins, temperature


def kl_metric(
    items: List[Dict[str, Any]],
    label_dict: Dict[str, int],
    exclusive: bool = False,
    use_ts: bool = False,
    temperature: float = 1.0,
) -> Tuple[List[float], float, float]:
    vals: List[float] = []
    for it in items:
        probs = softmax_probs(it["model_all"], temperature if use_ts else 1.0)
        if exclusive:
            keys = [k for k in it["human_conf"].keys() if k in label_dict]
            if len(keys) <= 1:
                continue
            p = np.array([float(it["human_conf"][k]) for k in keys], dtype=np.float64)
            q = np.array([float(probs[label_dict[k]]) for k in keys], dtype=np.float64)
            p = normalize(p + EPS)
            q = normalize(q + EPS)
        else:
            L = max(len(probs), max(label_dict.values()) + 1 if label_dict else 0)
            p = np.full(L, 1.0 / L, dtype=np.float64)
            for k, v in it["human_conf"].items():
                if k in label_dict:
                    p[label_dict[k]] = float(v)
            p = normalize(p + EPS)
            q = probs
        vals.append(float(kl_div(p, q)))

    if len(vals) == 0:
        return [], 0.0, temperature
    return vals, float(np.mean(vals)), temperature


def ent_ce(
    items: List[Dict[str, Any]],
    label_dict: Dict[str, int],
    exclusive: bool = False,
    use_ts: bool = False,
    temperature: float = 1.0,
) -> Tuple[List[float], List[float], float, float]:
    model_ents: List[float] = []
    human_ents: List[float] = []
    errs: List[float] = []

    for it in items:
        probs = softmax_probs(it["model_all"], temperature if use_ts else 1.0)
        if exclusive:
            keys = [k for k in it["human_conf"].keys() if k in label_dict]
            if len(keys) <= 1:
                continue
            p = np.array([float(it["human_conf"][k]) for k in keys], dtype=np.float64)
            q = np.array([float(probs[label_dict[k]]) for k in keys], dtype=np.float64)
            p = normalize(p + EPS)
            q = normalize(q + EPS)
        else:
            L = max(len(probs), max(label_dict.values()) + 1 if label_dict else 0)
            p = np.full(L, 1.0 / L, dtype=np.float64)
            for k, v in it["human_conf"].items():
                if k in label_dict:
                    p[label_dict[k]] = float(v)
            p = normalize(p + EPS)
            q = probs
        he = float(kl_div(p, np.full_like(p, 1.0 / len(p))) + 0.0)  # not used; placeholder if needed
        me = float(kl_div(q, np.full_like(q, 1.0 / len(q))) + 0.0)  # not used; placeholder if needed
        he = float(-(p * np.log(p + EPS)).sum())
        me = float(-(q * np.log(q + EPS)).sum())
        model_ents.append(me)
        human_ents.append(he)
        errs.append(abs(me - he))

    if len(errs) == 0:
        return [], [], 0.0, temperature
    return model_ents, human_ents, float(np.mean(errs)), temperature


def tvd_metric(
    items: List[Dict[str, Any]],
    label_dict: Dict[str, int],
    exclusive: bool = False,
    use_ts: bool = False,
    temperature: float = 1.0,
) -> Tuple[List[float], float, float]:
    vals: List[float] = []
    for it in items:
        probs = softmax_probs(it["model_all"], temperature if use_ts else 1.0)
        if exclusive:
            keys = [k for k in it["human_conf"].keys() if k in label_dict]
            if len(keys) <= 1:
                continue
            p = np.array([float(it["human_conf"][k]) for k in keys], dtype=np.float64)
            q = np.array([float(probs[label_dict[k]]) for k in keys], dtype=np.float64)
            p = normalize(p + EPS)
            q = normalize(q + EPS)
        else:
            L = max(len(probs), max(label_dict.values()) + 1 if label_dict else 0)
            p = np.full(L, 1.0 / L, dtype=np.float64)
            for k, v in it["human_conf"].items():
                if k in label_dict:
                    p[label_dict[k]] = float(v)
            p = normalize(p + EPS)
            q = probs
        vals.append(float(np.sum(np.abs(p - q)) / 2.0))
    if len(vals) == 0:
        return [], 0.0, temperature
    return vals, float(np.mean(vals)), temperature


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--beit_labelpath", default="/nfs/gdata/jian/project/vqa/emnlp24/models/unilm/beit3/data/vqa2.0/answer2label.txt")
    p.add_argument("--lxmert_labelpath", default="/nfs/gdata/jian/project/vqa/emnlp24/models/lxmert-master/data/vqa/trainval_ans2label.json")
    p.add_argument("--pre_path", default="/nfs/gdata/user/project/vqa/conf24/models/unilm/beit3/eval_scores/beit_low.json")
    p.add_argument("--annotations", default="/nfs/gdata/user/project/vqa/conf24/models/unilm/beit3/data/vqa2.0/vqa/v2_mscoco_val2014_annotations.json")
    p.add_argument("--use_label", choices=["beit", "lxmert"], default="beit")
    p.add_argument("--ts", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--exclusive", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    beit_labels = load_beit_label_dict(Path(args.beit_labelpath))
    lxmert_labels = load_lxmert_label_dict(Path(args.lxmert_labelpath))
    label_dict = beit_labels if args.use_label == "beit" else lxmert_labels

    items = load_json(Path(args.pre_path))

    ece, cmin, cmax, rt_acc, rt_confs, rt_bin_count, n_bins, temp = ece_score(
        items,
        n_bins=args.n_bins,
        label_dict=label_dict,
        annotations_path=Path(args.annotations),
        use_ts=args.ts,
        temperature=args.temperature,
    )
    print("ECE:", ece, "min_conf:", cmin, "max_conf:", cmax, "bins:", n_bins, "T:", temp)
    print("BinAcc:", rt_acc)
    print("BinConfs:", rt_confs)
    print("BinCount:", rt_bin_count)

    tvd_list, tvd_mean, temp = tvd_metric(
        items,
        label_dict=label_dict,
        exclusive=args.exclusive,
        use_ts=args.ts,
        temperature=args.temperature,
    )
    print("TVD:", tvd_mean, "in_use:", len(tvd_list), "T:", temp)

    kl_list, kl_mean, temp = kl_metric(
        items,
        label_dict=label_dict,
        exclusive=args.exclusive,
        use_ts=args.ts,
        temperature=args.temperature,
    )
    print("KL:", kl_mean, "in_use:", len(kl_list), "T:", temp)

    model_ent, human_ent, entce_mean, temp = ent_ce(
        items,
        label_dict=label_dict,
        exclusive=args.exclusive,
        use_ts=args.ts,
        temperature=args.temperature,
    )
    print("EntCE:", entce_mean, "in_use:", len(model_ent), "T:", temp)
