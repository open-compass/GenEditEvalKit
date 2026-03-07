import argparse
import csv
import os
from typing import Dict, List, Tuple, Set, Optional

ALL_METRICS: List[str] = [
    "detail_preserving",
    "instruction_following",
    "visual_quality",
    "knowledge_fidelity",
    "creative_fusion",
]

METRIC_SCORE_KEYS: Dict[str, str] = {
    "detail_preserving": "DP_score",
    "instruction_following": "IF_score",
    "visual_quality": "VQ_score",
    "knowledge_fidelity": "KF_score",
    "creative_fusion": "CF_score",
}

CATEGORY_METRICS: Dict[str, List[str]] = {
    "Imagination": ["detail_preserving", "instruction_following", "visual_quality", "creative_fusion"],
    "Awareness": ["detail_preserving", "instruction_following", "visual_quality", "knowledge_fidelity"],
    "Interpretation": ["detail_preserving", "instruction_following", "visual_quality", "knowledge_fidelity"],
    "WiseEdit_Complex": ["detail_preserving", "instruction_following", "visual_quality", "knowledge_fidelity", "creative_fusion"],
}

METRIC_ABBR: Dict[str, str] = {
    "detail_preserving": "DF",
    "instruction_following": "IF",
    "visual_quality": "VQ",
    "knowledge_fidelity": "KF",
    "creative_fusion": "CF",
}


def scale_score(raw: float) -> float:
    """Map raw score (1–10, or possibly 0) to [0, 100]"""
    if raw is None:
        return 0.0
    scaled = (raw - 1.0) / 9.0 * 100.0
    return max(scaled, 0.0)


def list_base_subsets_by_category(dataset_dir: str) -> Dict[str, List[str]]:
    ret: Dict[str, List[str]] = {k: [] for k in CATEGORY_METRICS.keys()}

    wiseedit_dir = os.path.join(dataset_dir, "WiseEdit")
    if not os.path.isdir(wiseedit_dir):
        raise RuntimeError(f"WiseEdit directory not found under dataset_dir: {wiseedit_dir}")

    for cat in ("Imagination", "Awareness", "Interpretation"):
        cat_dir = os.path.join(wiseedit_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for subset in os.listdir(cat_dir):
            subset_dir = os.path.join(cat_dir, subset)
            if not os.path.isdir(subset_dir):
                continue
            csv_path = os.path.join(subset_dir, f"{subset}.csv")
            if os.path.isfile(csv_path):
                ret[cat].append(subset)

    complex_root = os.path.join(dataset_dir, "WiseEdit-Complex")
    if os.path.isdir(complex_root):
        for subset in os.listdir(complex_root):
            subset_dir = os.path.join(complex_root, subset)
            if not os.path.isdir(subset_dir):
                continue
            csv_path = os.path.join(subset_dir, f"{subset}.csv")
            if os.path.isfile(csv_path):
                ret["WiseEdit_Complex"].append(subset)

    for k in ret:
        ret[k] = sorted(ret[k])

    return ret


def get_base_csv_path(dataset_dir: str, cat: str, subset: str) -> str:
    if cat in ("Imagination", "Awareness", "Interpretation"):
        wiseedit_dir = os.path.join(dataset_dir, "WiseEdit")
        return os.path.join(wiseedit_dir, cat, subset, f"{subset}.csv")
    elif cat == "WiseEdit_Complex":
        complex_root = os.path.join(dataset_dir, "WiseEdit-Complex")
        return os.path.join(complex_root, subset, f"{subset}.csv")
    else:
        raise ValueError(f"Unknown category: {cat}")


def load_idx_set_from_csv(csv_path: str) -> Set[str]:
    idx_set: Set[str] = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_val = row.get("idx") or row.get("\ufeffidx")
            if idx_val is None:
                continue
            idx_str = str(idx_val).strip()
            if idx_str:
                idx_set.add(idx_str)
    return idx_set


def load_score_rows_and_idx(score_csv_path: str) -> Tuple[List[Dict[str, str]], Set[str]]:
    rows: List[Dict[str, str]] = []
    idx_set: Set[str] = set()
    with open(score_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            idx_val = row.get("idx") or row.get("\ufeffidx")
            if idx_val is None:
                continue
            idx_str = str(idx_val).strip()
            if idx_str:
                idx_set.add(idx_str)
    return rows, idx_set


def safe_parse_score(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None
    try:
        v = int(txt)
    except Exception:
        return 1
    return v if v > 0 else None


def row_has_zero_or_empty(row: Dict[str, str], metrics: List[str]) -> bool:
    for m in metrics:
        base_key = METRIC_SCORE_KEYS[m]
        for lang in ("cn", "en"):
            col = f"{base_key}_{lang}"
            val = row.get(col)
            if val is None:
                return True
            s = str(val).strip()
            if s == "" or s == "0":
                return True
    return False


def summarize_one_model_by_category(
        model_tag: str,
        base_subsets: Dict[str, List[str]],
        dataset_dir: str,
        score_root: str,
) -> Optional[Dict[str, float]]:
    model_score_dir = os.path.join(score_root, model_tag)
    if not os.path.isdir(model_score_dir):
        print(f"[{model_tag}] score directory not found: {model_score_dir}")
        return None

    sums: Dict[Tuple[str, str, str], float] = {}
    counts: Dict[Tuple[str, str, str], int] = {}
    for cat, metrics in CATEGORY_METRICS.items():
        for lang in ("cn", "en"):
            for m in metrics:
                sums[(cat, lang, m)] = 0.0
                counts[(cat, lang, m)] = 0

    for cat, subsets in base_subsets.items():
        metrics_for_cat = CATEGORY_METRICS[cat]
        for subset in subsets:
            # base_csv_name = subset + ".csv"
            # base_csv_path = os.path.join(dataset_dir, base_csv_name)
            base_csv_path = get_base_csv_path(dataset_dir, cat, subset)
            if not os.path.isfile(base_csv_path):
                print(f"[{model_tag}] base csv missing in listing: {base_csv_path}")
                continue

            score_fname = f"score_{subset}.csv"
            score_csv_path = os.path.join(model_score_dir, score_fname)

            if os.path.isfile(score_csv_path):
                base_idx = load_idx_set_from_csv(base_csv_path)
                score_rows, score_idx = load_score_rows_and_idx(score_csv_path)

                if base_idx != score_idx:
                    missing_in_score = base_idx - score_idx
                    extra_in_score = score_idx - base_idx
                    raise RuntimeError(
                        f"[{model_tag}] idx mismatch between base and score for subset '{subset}'.\n"
                        f"  base CSV: {base_csv_path}\n"
                        f"  score CSV: {score_csv_path}\n"
                        f"  base_idx count={len(base_idx)}, score_idx count={len(score_idx)}\n"
                        f"  missing in score (first 10): {list(missing_in_score)[:10]}\n"
                        f"  extra in score (first 10): {list(extra_in_score)[:10]}"
                    )

                kept = 0
                skipped = 0
                for row in score_rows:
                    if row_has_zero_or_empty(row, metrics_for_cat):
                        skipped += 1
                        continue
                    kept += 1
                    for m in metrics_for_cat:
                        base_key = METRIC_SCORE_KEYS[m]
                        for lang in ("cn", "en"):
                            col = f"{base_key}_{lang}"
                            v = safe_parse_score(row.get(col))
                            if v is None:
                                continue
                            sums[(cat, lang, m)] += v
                            counts[(cat, lang, m)] += 1
                print(f"[{model_tag}] {subset}: kept={kept}, skipped={skipped}")

            else:  # If the model does not have a corresponding score_*.csv file, all values will be recorded as 1 for calculation.
                num_rows = len(load_idx_set_from_csv(base_csv_path))
                if num_rows <= 0:
                    print(f"[{model_tag}] subset {subset} base has 0 rows, skip filling.")
                    continue
                for m in metrics_for_cat:
                    for lang in ("cn", "en"):
                        sums[(cat, lang, m)] += 1.0 * num_rows
                        counts[(cat, lang, m)] += num_rows
                print(f"[{model_tag}] {subset}: score missing -> filled ones, rows={num_rows}")

    result: Dict[str, float] = {}

    def compute_cat_lang_means(cat: str, lang: str) -> Tuple[Dict[str, float], float]:
        metric_means: Dict[str, float] = {}
        metrics = CATEGORY_METRICS[cat]
        vals_for_overall: List[float] = []
        for m in metrics:
            c = counts[(cat, lang, m)]
            mean_raw = sums[(cat, lang, m)] / c
            scaled = scale_score(mean_raw)
            mean_v = round(scaled, 1)
            metric_means[m] = mean_v
            vals_for_overall.append(mean_v)
        overall = sum(vals_for_overall) / len(vals_for_overall) if vals_for_overall else 0.0
        return metric_means, overall

    for cat in ("Imagination", "Awareness", "Interpretation", "WiseEdit_Complex"):
        for lang in ("cn", "en"):
            metric_means, overall = compute_cat_lang_means(cat, lang)
            for m, v in metric_means.items():
                result[f"{cat}_{m}_{lang}"] = v
            result[f"{cat}_overall_{lang}"] = overall

    for lang in ("cn", "en"):
        basics = [
            result[f"Imagination_overall_{lang}"],
            result[f"Awareness_overall_{lang}"],
            result[f"Interpretation_overall_{lang}"],
        ]
        result[f"basic_overall_{lang}"] = sum(basics) / len(basics)

    cx_cn = result.get("WiseEdit_Complex_overall_cn", 0.0)
    cx_en = result.get("WiseEdit_Complex_overall_en", 0.0)
    result["WiseEdit_Complex_overall"] = (cx_cn + cx_en) / 2.0

    return result


def build_headers():
    header_cn: List[str] = ["model"]
    header_en: List[str] = ["model"]
    header_complex: List[str] = ["model"]

    for cat in ("Imagination", "Awareness", "Interpretation"):
        for m in CATEGORY_METRICS[cat]:
            header_cn.append(f"{cat}_{m}_cn")
            header_en.append(f"{cat}_{m}_en")
        header_cn.append(f"{cat}_overall_cn")
        header_en.append(f"{cat}_overall_en")

    header_cn.append("basic_overall_cn")
    header_en.append("basic_overall_en")

    for m in CATEGORY_METRICS["WiseEdit_Complex"]:
        header_complex.append(f"WiseEdit_Complex_{m}_cn")
    header_complex.append("WiseEdit_Complex_overall_cn")
    for m in CATEGORY_METRICS["WiseEdit_Complex"]:
        header_complex.append(f"WiseEdit_Complex_{m}_en")
    header_complex.append("WiseEdit_Complex_overall_en")

    header_complex.append("WiseEdit_Complex_overall")

    return header_cn, header_en, header_complex


def print_final_results(model_tag: str, summary: Dict[str, float]) -> None:
    print()
    print(f"------------------------- Final Result of {model_tag} -------------------------")

    for lang, lang_label in (("cn", "Chinese"), ("en", "English")):
        print(f"\nWiseEdit-{lang_label} version")
        task_order = ["Awareness", "Interpretation", "Imagination", "WiseEdit_Complex"]
        for idx, cat in enumerate(task_order, start=1):
            display_name = cat
            metrics = CATEGORY_METRICS[cat]
            parts = [f"Task{idx}: {display_name}"]
            for m in metrics:
                key = f"{cat}_{m}_{lang}"
                val = summary.get(key, 0.0)
                abbr = METRIC_ABBR.get(m, m)
                parts.append(f"{abbr}: {val:.1f}")
            overall_key = f"{cat}_overall_{lang}"
            overall_val = summary.get(overall_key, 0.0)
            parts.append(f"AVG: {overall_val:.1f}")
            print("  " + "  ".join(parts))

        basic_overall = summary.get(f"basic_overall_{lang}", 0.0)
        print(f"  Basic Overall AVG: {basic_overall:.1f}")

    complex_overall = summary.get("WiseEdit_Complex_overall", 0.0)
    print(f"\nComplex Overall (CN+EN AVG): {complex_overall:.1f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate WiseEdit evaluation results for a single model."
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to WiseEdit-Benchmark.")
    parser.add_argument("--score_root",  type=str, required=True, help="Root directory that contains model score_*.csv folders.")
    parser.add_argument("--name",   type=str, required=True, help="Model tag, i.e., subfolder name under score_root.")
    parser.add_argument("--statistic_output_dir",  type=str, default=None, help="Directory to save summary CSVs; defaults to score_root if not set.")
    parser.add_argument("--regenerate",  action="store_true", help="Overwrite existing summary CSVs if they already exist.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.statistic_output_dir is None:
        args.statistic_output_dir = args.score_root

    print(f"[INFO] dataset_dir  = {args.dataset_dir}")
    print(f"[INFO] score_root   = {args.score_root}")
    print(f"[INFO] model_name   = {args.name}")
    print(f"[INFO] output_dir   = {args.statistic_output_dir}")

    base_subsets = list_base_subsets_by_category(args.dataset_dir)
    header_cn, header_en, header_complex = build_headers()

    summary = summarize_one_model_by_category(
        model_tag=args.name,
        base_subsets=base_subsets,
        dataset_dir=args.dataset_dir,
        score_root=args.score_root,
    )

    if summary is None:
        print(f"[ERROR] summarize_one_model_by_category returned None for model {args.name}")
        return

    row_cn: Dict[str, object] = {"model": args.name}
    row_en: Dict[str, object] = {"model": args.name}
    row_cx: Dict[str, object] = {"model": args.name}
    for col in header_cn:
        if col == "model":
            continue
        row_cn[col] = summary.get(col, float("nan"))
    for col in header_en:
        if col == "model":
            continue
        row_en[col] = summary.get(col, float("nan"))
    for col in header_complex:
        if col == "model":
            continue
        row_cx[col] = summary.get(col, float("nan"))

    os.makedirs(args.statistic_output_dir, exist_ok=True)
    summary_cn_csv = os.path.join(args.statistic_output_dir, f"{args.name}_cn.csv")
    summary_en_csv = os.path.join(args.statistic_output_dir, f"{args.name}_en.csv")
    summary_complex_csv = os.path.join(args.statistic_output_dir, f"{args.name}_complex.csv")

    if (not args.regenerate) and all(
            os.path.exists(p) for p in (summary_cn_csv, summary_en_csv, summary_complex_csv)
    ):
        print(
            f"[INFO] Summary CSVs already exist for model '{args.name}' in {args.statistic_output_dir}. "
            f"Use --regenerate to overwrite."
        )
    else:
        with open(summary_cn_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_cn)
            writer.writeheader()
            writer.writerow(row_cn)
        print(f"[INFO] Wrote CN summary: {summary_cn_csv}")

        with open(summary_en_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_en)
            writer.writeheader()
            writer.writerow(row_en)
        print(f"[INFO] Wrote EN summary: {summary_en_csv}")

        with open(summary_complex_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_complex)
            writer.writeheader()
            writer.writerow(row_cx)
        print(f"[INFO] Wrote COMPLEX summary: {summary_complex_csv}")

    print_final_results(args.name, summary)


if __name__ == "__main__":
    main()