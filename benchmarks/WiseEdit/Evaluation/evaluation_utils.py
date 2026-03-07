import os
import json
import base64
import time
import re
import logging
from typing import List, Optional, Dict, Tuple
from PIL import Image
from .prompt_single import *
from .prompt_multi import *
import io

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

ALL_METRICS = [
    "detail_preserving",
    "instruction_following",
    "visual_quality",
    "knowledge_fidelity",
    "creative_fusion",
]

DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def init_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        return OpenAI(api_key=api_key)


def get_metric_prompt(metric: str, is_multi_input: bool = False) -> str:
    if metric == "visual_quality":
        return prompt_visual_quality

    if not is_multi_input:
        if metric == "detail_preserving":
            return prompt_detail_preserving_single
        elif metric == "instruction_following":
            return prompt_instruction_following_single
        elif metric == "knowledge_fidelity":
            return prompt_knowledge_fidelity_single
        elif metric == "creative_fusion":
            return prompt_creative_fusion_single
    else:
        if metric == "detail_preserving":
            return prompt_detail_preserving_multi
        elif metric == "instruction_following":
            return prompt_instruction_following_multi
        elif metric == "knowledge_fidelity":
            return prompt_knowledge_fidelity_multi
        elif metric == "creative_fusion":
            return prompt_creative_fusion_multi

    return ""


def encode_image_to_base64(path: str) -> Optional[str]:
    # resize to 512*512
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((512, 512))
            # logging.info(f"Thumbnail created for {os.path.basename(path)} with size {img.size}")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        logging.error(f"Failed to encode image {path}: {e}")
        return None


def extract_json_field(response: str) -> Tuple[Optional[int], Optional[str]]:
    """
    The unified output format is: {"score": 8, "reason": "..."} — where score is on a 1–10 scale.
    """
    pattern = r"\{[^{}]*score[^{}]*\}"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None, None

    snippet = match.group(0)
    try:
        data = json.loads(snippet)
        score = data.get("score")
        reason = data.get("reason")
        if score is not None:
            score = int(score)
        return score, reason
    except Exception:
        return None, None


DEFAULT_PATTERNS = [
    r"['\"]?\s*score\s*['\"]?\s*[:：]\s*(10|[1-9])\b",
    r"score\s*[:：]\s*(10|[1-9])\b",
    r"rating\s*[:：]\s*(10|[1-9])\b",
    r"(10|[1-9])\s*/\s*10",
    r"(10|[1-9])\s+out\s+of\s+10",
    r"\b(10|[1-9])\b",
]


def extract_score_and_reason_generic(
    response: str,
) -> Tuple[Optional[int], Optional[str]]:

    score, reason = extract_json_field(response)
    if score is not None:
        if not (1 <= score <= 10):
            score = None
        return score, reason

    for pat in DEFAULT_PATTERNS:
        m = re.search(pat, response, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                s = int(m.group(1))
                return s, None
            except Exception:
                continue
    return None, None


def build_message_for_metric(
    metric: str,
    instruction: str,
    input_images_b64: List[str],
    is_multi_input: bool,
    edited_image_b64: str,
    hint: Optional[str] = None,
    ref_images_b64: Optional[List[str]] = None,
) -> dict:

    base_prompt = get_metric_prompt(metric,is_multi_input)
    if not base_prompt:
        base_prompt = (
            "You are an expert image editing evaluator.\n"
            f"Please evaluate the '{metric}' score on a scale from 1 to 10 "
            "(1 = very bad, 10 = excellent) for the edited image, "
            "given the editing instruction and the provided images.\n"
            "Respond only with a short JSON object like: "
            "{\"score\": 8, \"reason\": \"...\"}."
        )

    if metric == "visual_quality":
        content = [
            {"type": "text", "text": base_prompt},
            {"type": "text", "text": "Image:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{edited_image_b64}"}
            },
        ]
        return {"role": "user", "content": content}

    content = []
    full_text = (
            base_prompt + "\n\n"
            f"Editing instruction: {instruction}"
    )
    content.append({"type": "text", "text": full_text})

    if input_images_b64:
        if is_multi_input:
            for idx, img_b64 in enumerate(input_images_b64, start=1):
                content.append({"type": "text", "text": f"Input image #{idx}:"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
        else:
            content.append({"type": "text", "text": "Input image:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{input_images_b64[0]}"}
            })

    content.append({"type": "text", "text": "Edited image:"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{edited_image_b64}"}
    })

    use_hint_and_ref = metric in ("instruction_following", "knowledge_fidelity")
    if use_hint_and_ref and hint:
        content.append({
            "type": "text",
            "text": (
                "Hint (important for scoring): "
                "The following hint describes the correct expected result of the editing task. "
                "Use this as a direct reference for judging how well the edited image meets the intended goal.\n"
                f"{hint}"
            ),
        })

    if use_hint_and_ref and ref_images_b64:
        for idx, ref_b64 in enumerate(ref_images_b64, start=1):
            content.append({"type": "text", "text": f"Reference image:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}
            })

    message = {
        "role": "user",
        "content": content,
    }
    return message


def call_gpt_with_retry(
    message: dict,
    metric: str,
    max_retries: int = 3,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str = DEFAULT_API_KEY,
    base_url: str = DEFAULT_BASE_URL
) -> Tuple[int, Optional[str]]:
    client = init_client(api_key, base_url)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[message],
                max_tokens=1000,
                stream=False,
            )
            text_resp = resp.choices[0].message.content
            # print(text_resp) # test
            score, reason = extract_score_and_reason_generic(text_resp)
            if score is not None:
                return score, reason
            logging.warning(
                f"[{metric}] Parsed score is None on attempt {attempt}/{max_retries}, will retry."
            )

        except Exception as e:
            logging.warning(
                f"[{metric}] GPT call failed on attempt {attempt}/{max_retries}: {e}"
            )

        time.sleep(5)

    logging.error(f"important error {model_name}: [{metric}] Failed after {max_retries} attempts, using score=0.")
    return 0, None


def evaluate_example_with_gpt(
    input_image_paths: List[str],
    is_multi_input: bool,
    edited_image_path: str,
    instruction: str,
    metrics: List[str],
    hint: Optional[str] = None,
    ref_image_paths: Optional[List[str]] = None,
    max_retries: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str = DEFAULT_API_KEY,
    base_url: str = DEFAULT_BASE_URL
) -> Dict[str, Optional[int]]:
    """
    Evaluate a single example with GPT and return scores (1–10) for the requested metrics.

    Args:
        input_image_paths: Paths to the original input images (one or more).
        is_multi_input: Whether to treat this as a multi-image task (images labeled #1, #2, ... in the prompt).
        edited_image_path: Path to the edited image.
        instruction: Text editing instruction.
        metrics: Subset of metrics to evaluate, e.g. ["detail_preserving", "visual_quality"].
        hint: Optional textual hint about the expected result.
        ref_image_paths: Optional paths to reference images.
        max_retries: Maximum retries when the model call or parsing fails.
        model_name: Model name for GPT evaluation.
        api_key: API key for the OpenAI client.
        base_url: Optional custom API base URL.

    Returns:
        A dict mapping each metric in ALL_METRICS to an int or None:
          - In metrics and succeeded: 1–10
          - In metrics but failed after retries: 0
          - Not in metrics: None
    """

    scores: Dict[str, Optional[int]] = {m: None for m in ALL_METRICS}

    input_images_b64: List[str] = []
    for p in input_image_paths:
        b64 = encode_image_to_base64(p)
        if b64:
            input_images_b64.append(b64)
        else:
            logging.warning(f"Skip bad input image: {p}")

    edited_b64 = encode_image_to_base64(edited_image_path)
    if not edited_b64:
        logging.error(f"Cannot encode edited image: {edited_image_path}. All metrics set to 0.")
        for m in metrics:
            if m in ALL_METRICS:
                scores[m] = 0
        return scores

    ref_images_b64: List[str] = []
    if ref_image_paths:
        for p in ref_image_paths:
            b64 = encode_image_to_base64(p)
            if b64:
                ref_images_b64.append(b64)
            else:
                logging.warning(f"Skip bad ref image: {p}")

    for metric in metrics:
        if metric not in ALL_METRICS:
            logging.warning(f"Unknown metric '{metric}', skip.")
            continue

        message = build_message_for_metric(
            metric=metric,
            instruction=instruction,
            input_images_b64=input_images_b64,
            is_multi_input=is_multi_input,
            edited_image_b64=edited_b64,
            hint=hint,
            ref_images_b64=ref_images_b64,
        )
        score, _reason = call_gpt_with_retry(message, metric, max_retries=max_retries,model_name=model_name,api_key=api_key,base_url=base_url)
        scores[metric] = score

    return scores
