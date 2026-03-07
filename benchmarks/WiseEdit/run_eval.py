import os
import sys
import csv
import json
import logging
import argparse
from typing import List, Optional, Dict, Tuple
from concurrent.futures import as_completed, ThreadPoolExecutor
from Evaluation.evaluation_utils import evaluate_example_with_gpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

TARGET_CSV_FILES: List[str] = []

default_CSV_files: List[str] = [
    "Imagination_1.csv",
    "Imagination_2.csv",
    "Imagination_3.csv",
    "Imagination_4.csv",
    "Imagination_5.csv",
    "Awareness_1.csv",
    "Awareness_2.csv",
    "Interpretation_1.csv",
    "WiseEdit_Complex_2.csv",
    "WiseEdit_Complex_3.csv",
    "WiseEdit_Complex_4.csv",
]

ALL_METRICS = [
    "detail_preserving",
    "instruction_following",
    "visual_quality",
    "knowledge_fidelity",
    "creative_fusion",
]  # all metrics could be used to eval

METRIC_SCORE_KEYS = {
    "detail_preserving": "DP_score",
    "instruction_following": "IF_score",
    "visual_quality": "VQ_score",
    "knowledge_fidelity": "KF_score",
    "creative_fusion": "CF_score",
}  # print out word

CSV_METRICS = {
    "Imagination_1.csv": [
        "detail_preserving",
        "instruction_following",
        "visual_quality",
        "creative_fusion",
    ],
    "Imagination_2.csv": [
        "detail_preserving",
        "instruction_following",
        "visual_quality",
        "creative_fusion",
    ],
    "Imagination_3.csv": [
        "detail_preserving",
        "instruction_following",
        "visual_quality",
        "creative_fusion",
    ],
    "Imagination_4.csv": [
        "detail_preserving",
        "instruction_following",
        "visual_quality",
        "creative_fusion",
    ],
    "Imagination_5.csv": [
        "detail_preserving",
        "instruction_following",
        "visual_quality",
        "creative_fusion",
    ],
    "Awareness_1.csv": [
        "detail_preserving",
        "instruction_following",
        "visual_quality",
        "knowledge_fidelity",
    ],
    "Awareness_2.csv": [
        "detail_preserving",
        "instruction_following",
        "visual_quality",
        "knowledge_fidelity",
    ],
    "Interpretation_1.csv": [
        "detail_preserving",
        "instruction_following",
        "visual_quality",
        "knowledge_fidelity",
    ],
}   # different metrics to different task, no setting in this will use default metrics(all metrics)
DEFAULT_METRICS = ALL_METRICS


# =====================================================

def find_edited_image(subset_name: str, lang: str, idx: str, result_img_root: Optional[str] = None,) -> Optional[str]:
    folder = os.path.join(result_img_root, subset_name, lang)
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = os.path.join(folder, f"{idx}{ext}")
        if os.path.exists(p):
            return p
    logging.error(f"[{subset_name}] {lang} result image for idx={idx} not found in {folder}")
    return None


def collect_input_images(row: dict, num_inputs: Optional[int], dataset_root: Optional[str] = None,) -> Tuple[List[str], bool]:
    paths: List[str] = []

    if num_inputs is not None:
        for i in range(1, num_inputs + 1):
            key = f"input_{i}"
            v = row.get(key, "")
            if v:
                v = v.strip()
                if v:
                    paths.append(v)
    else:
        for key, v in row.items():
            if key.startswith("input_") and isinstance(v, str):
                v = v.strip()
                if v:
                    paths.append(v)

    if dataset_root:
        paths = [os.path.join(dataset_root, p) for p in paths]

    is_multi = len(paths) > 1
    return paths, is_multi


def _row_is_fully_scored(
    row: dict,
    metrics_to_eval: List[str],
) -> bool:
    for m in metrics_to_eval:
        base_key = METRIC_SCORE_KEYS[m]
        for lang in ("cn", "en"):
            col = f"{base_key}_{lang}"
            val = row.get(col, "")
            if val is None:
                return False
            if isinstance(val, str):
                v = val.strip()
            else:
                v = str(val).strip()
            if not v:
                return False
            try:
                int(v)
            except Exception:
                return False
    return True


def run_eval_for_one_csv(
    csv_path: str,
    max_workers: int = 5,
    model_name: str = None,
    api_key: str = None,
    base_url: str = None,
    model_tag: str = None,
    result_img_root: Optional[str] = None,
    score_output_root: Optional[str] = None,
    dataset_root: str = None,
) -> None:
    """
    Multi-thread evaluate one CSV file and save results to score_<subset_name>.csv.
    """
    subset_name = os.path.splitext(os.path.basename(csv_path))[0]
    csv_filename = os.path.basename(csv_path)

    num_inputs: Optional[int] = None
    last_part = subset_name.split("_")[-1]
    if last_part.isdigit():
        num_inputs = int(last_part)

    metrics_to_eval = CSV_METRICS.get(csv_filename, DEFAULT_METRICS)
    subset_dir = os.path.join(result_img_root, subset_name)
    if not os.path.exists(subset_dir):
        logging.warning(
            f"[{subset_name}] Skipped — result directory not found: {subset_dir}"
        )
        return

    logging.info(
        f"Start evaluating CSV: {csv_path} "
        f"(subset={subset_name}, model={model_tag}, num_inputs={num_inputs}, metrics={metrics_to_eval})"
    )

    os.makedirs(score_output_root, exist_ok=True)
    out_csv_path = os.path.join(
        score_output_root,
        f"score_{subset_name}.csv"
    )

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    score_fields_cn: List[str] = []
    score_fields_en: List[str] = []
    for m in ALL_METRICS:
        base_key = METRIC_SCORE_KEYS[m]
        score_fields_cn.append(f"{base_key}_cn")
        score_fields_en.append(f"{base_key}_en")

    out_fieldnames = fieldnames + score_fields_cn + score_fields_en
    processed_idx: set[str] = set()
    existing_rows_by_idx: Dict[str, dict] = {}

    if os.path.exists(out_csv_path):
        logging.info(f"[{subset_name}] Found existing score file, will reuse: {out_csv_path}")
        with open(out_csv_path, "r", encoding="utf-8-sig", newline="") as f_exist:
            exist_reader = csv.DictReader(f_exist)
            for erow in exist_reader:
                eidx = erow.get("idx") or erow.get("\ufeffidx")
                if eidx is None:
                    continue
                idx_str = str(eidx).strip()
                if not idx_str:
                    continue
                existing_rows_by_idx[idx_str] = erow

        for idx_str, erow in existing_rows_by_idx.items():
            if _row_is_fully_scored(erow, metrics_to_eval):
                processed_idx.add(idx_str)

        logging.info(
            f"[{subset_name}] existing score file rows = {len(existing_rows_by_idx)}, "
            f"fully-scored idx count = {len(processed_idx)}"
        )
    else:
        logging.info(f"[{subset_name}] No existing score file, start fresh.")

    to_eval_rows: List[Tuple[str, dict]] = []
    idx_order: Dict[str, int] = {}

    for order, row in enumerate(rows):
        idx_val = row.get("idx") or row.get("\ufeffidx")
        if idx_val is None:
            logging.warning("Found row with empty idx, skip.")
            continue
        idx_str = str(idx_val).strip()
        if not idx_str:
            logging.warning("Found row with empty idx (after strip), skip.")
            continue

        idx_order[idx_str] = order
        if idx_str in processed_idx:
            continue

        to_eval_rows.append((idx_str, row))

    logging.info(
        f"[{subset_name}] total rows = {len(rows)}, "
        f"need evaluation = {len(to_eval_rows)}"
    )

    if len(to_eval_rows) == 0 and os.path.exists(out_csv_path):
        logging.info(f"[{subset_name}] All rows already fully scored. Skip re-evaluation.")
        return

    # Multi-thread evaluation: each thread handles one row
    def eval_single_row(idx_str: str, row: dict, dataset_root: str) -> Tuple[str, Dict[str, Optional[int]], Dict[str, Optional[int]]]:
        scores_cn: Dict[str, Optional[int]] = {m: None for m in ALL_METRICS}
        scores_en: Dict[str, Optional[int]] = {m: None for m in ALL_METRICS}
        input_paths, is_multi = collect_input_images(row, num_inputs, dataset_root)

        instr = row.get("prompt", "")
        hint = row.get("hint", "").strip() or None

        ref_raw = row.get("ref", "")
        ref_col = ref_raw.strip() if isinstance(ref_raw, str) else ""
        ref_paths: List[str] = []
        if ref_col:
            try:
                obj = json.loads(ref_col)
                if isinstance(obj, list):
                    ref_paths = [str(x).strip() for x in obj if str(x).strip()]
                elif isinstance(obj, str):
                    ref_paths = [obj.strip()]
            except Exception:
                for sep in ["|", ";"]:
                    if sep in ref_col:
                        ref_paths = [p.strip() for p in ref_col.split(sep) if p.strip()]
                        break
                if not ref_paths:
                    ref_paths = [ref_col]

        if dataset_root and ref_paths:
            ref_paths = [os.path.join(dataset_root, p) for p in ref_paths]

        edited_cn = find_edited_image(subset_name, "cn", idx_str, result_img_root=result_img_root)
        edited_en = find_edited_image(subset_name, "en", idx_str, result_img_root=result_img_root)

        if instr.strip() and metrics_to_eval:
            if edited_cn:
                try:
                    sc = evaluate_example_with_gpt(
                        input_image_paths=input_paths,
                        is_multi_input=is_multi,
                        edited_image_path=edited_cn,
                        instruction=instr,
                        metrics=metrics_to_eval,
                        hint=hint,
                        ref_image_paths=ref_paths if ref_paths else None,
                        model_name=model_name,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    for m in ALL_METRICS:
                        if m in sc:
                            scores_cn[m] = sc[m]
                except Exception as e:
                    logging.error(f"[{subset_name}] Error evaluating CN idx={idx_str}: {e}", exc_info=True)
            else:
                logging.warning(
                    f"[{subset_name}] no CN image for idx={idx_str}, "
                    f"set required CN metrics to 0."
                )
                for m in metrics_to_eval:
                    if m in ALL_METRICS:
                        scores_cn[m] = 0
        else:
            logging.warning(
                f"[{subset_name}] skip CN eval for idx={idx_str} "
                f"(no prompt or no metrics_to_eval)."
            )

        if instr.strip() and metrics_to_eval:
            if edited_en:
                try:
                    sc = evaluate_example_with_gpt(
                        input_image_paths=input_paths,
                        is_multi_input=is_multi,
                        edited_image_path=edited_en,
                        instruction=instr,
                        metrics=metrics_to_eval,
                        hint=hint,
                        ref_image_paths=ref_paths if ref_paths else None,
                        model_name=model_name,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    for m in ALL_METRICS:
                        if m in sc:
                            scores_en[m] = sc[m]
                except Exception as e:
                    logging.error(f"[{subset_name}] Error evaluating EN idx={idx_str}: {e}", exc_info=True)
            else:
                logging.warning(
                    f"[{subset_name}] no EN image for idx={idx_str}, "
                    f"set required EN metrics to 0."
                )
                for m in metrics_to_eval:
                    if m in ALL_METRICS:
                        scores_en[m] = 0
        else:
            logging.warning(
                f"[{subset_name}] skip EN eval for idx={idx_str} "
                f"(no prompt or no metrics_to_eval)."
            )

        cn_str = ", ".join(f"{m}={scores_cn[m]}" for m in ALL_METRICS)
        en_str = ", ".join(f"{m}={scores_en[m]}" for m in ALL_METRICS)
        logging.info(f"[{subset_name}] idx={idx_str} CN scores: {cn_str}")
        logging.info(f"[{subset_name}] idx={idx_str} EN scores: {en_str}")

        return idx_str, scores_cn, scores_en

    # Start the thread pool.
    new_scores_by_idx: Dict[str, Tuple[Dict[str, Optional[int]], Dict[str, Optional[int]]]] = {}

    if to_eval_rows:
        logging.info(f"[{subset_name}] Start ThreadPoolExecutor with max_workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx_str, row in to_eval_rows:
                fut = executor.submit(eval_single_row, idx_str, row, dataset_root)
                future_to_idx[fut] = idx_str

            for fut in as_completed(future_to_idx):
                idx_str = future_to_idx[fut]
                try:
                    idx_ret, scores_cn, scores_en = fut.result()
                    new_scores_by_idx[idx_ret] = (scores_cn, scores_en)
                except Exception as e:
                    logging.error(f"[{subset_name}] Failed processing idx={idx_str}: {e}", exc_info=True)

    final_rows: List[dict] = []

    for row in rows:
        idx_val = row.get("idx") or row.get("\ufeffidx")
        if idx_val is None:
            continue
        idx_str = str(idx_val).strip()
        if not idx_str:
            continue

        base_row = existing_rows_by_idx.get(idx_str, row.copy())

        for m in ALL_METRICS:
            base_key = METRIC_SCORE_KEYS[m]
            for lang in ("cn", "en"):
                col = f"{base_key}_{lang}"
                if col not in base_row:
                    base_row[col] = None

        if idx_str in new_scores_by_idx:
            scores_cn, scores_en = new_scores_by_idx[idx_str]
            for m in ALL_METRICS:
                base_key = METRIC_SCORE_KEYS[m]
                base_row[f"{base_key}_cn"] = scores_cn.get(m)
                base_row[f"{base_key}_en"] = scores_en.get(m)

        final_rows.append(base_row)

    with open(out_csv_path, "w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)

    logging.info(f"[{subset_name}] Done. Result written to: {out_csv_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Model tag used to name result directories.")
    parser.add_argument("--dataset_dir",     type=str,  required=True,  help="Path to WiseEdit-Benchmark.")
    parser.add_argument("--result_img_root", type=str,  required=True,  help="Root directory of result images (without model name).")
    parser.add_argument("--score_output_root", type=str,  required=True,  help="Root directory of output score CSVs (without model name).")
    parser.add_argument("--num_workers", type=int,  required=False, default=5, help="Number of threads per CSV.")
    parser.add_argument("--eval_model", type=str, required=False, default="gpt-4o", help="Model name used for scoring.")
    parser.add_argument("--target_csv", type=str, nargs="*", required=False, default=None,
                        help="Optional list of CSV file names to evaluate; if omitted, all CSVs in csv_dir are used.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    if not base_url:
        base_url = "https://api.openai.com/v1"
    if not api_key:
        logging.error("Environment variables API_KEY are not set; please run 'export API_KEY=your_key' in the terminal first.")
        sys.exit(1)

    # print(api_key)
    # print(base_url)

    model_tag = args.name
    eval_model = args.eval_model
    dataset_dir = args.dataset_dir

    result_img_root = os.path.join(args.result_img_root, model_tag)
    score_output_root = os.path.join(args.score_output_root, model_tag)
    os.makedirs(score_output_root, exist_ok=True)

    # add file logger
    log_file = os.path.join(score_output_root, f"{model_tag}_eval.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logging.info("=" * 120)
    logging.info(f"   Start evaluating model: {model_tag}")
    logging.info(f"   eval_model         = {eval_model}")
    logging.info(f"   dataset_dir        = {dataset_dir}")
    logging.info(f"   result_img_root    = {result_img_root}")
    logging.info(f"   score_output_root  = {score_output_root}")
    logging.info(f"   num_workers        = {args.num_workers}")
    logging.info("=" * 120)

    csv_files = []

    def _walk_csv_under(root_dir: str):
        for root, dirs, files in os.walk(root_dir):
            # Do NOT descend into 'imgs' or 'img_ref'
            dirs[:] = [d for d in dirs if d not in ("imgs", "img_ref")]
            for fn in files:
                if fn.lower().endswith(".csv"):
                    yield root, fn


    if args.target_csv:
        target_names = set(args.target_csv)
    else:
        target_names = set(default_CSV_files)

    for root, fn in _walk_csv_under(dataset_dir):
        if fn in target_names:
            csv_files.append(os.path.join(root, fn))

    if not csv_files:
        logging.error(f"There is no matching csv in: {dataset_dir}")
        sys.exit(1)

    for csv_path in csv_files:
        run_eval_for_one_csv(
            csv_path=csv_path,
            max_workers=args.num_workers,
            model_name=eval_model,
            api_key=api_key,
            base_url=base_url,
            model_tag=model_tag,
            result_img_root=result_img_root,
            score_output_root=score_output_root,
            dataset_root=dataset_dir,
        )

    logging.info("All CSVs finished for model: %s", model_tag)
    logging.info("Score result could be found in %s", score_output_root)