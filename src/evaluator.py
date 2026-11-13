from __future__ import annotations

import math
import re
from typing import Any, Optional

from src.schemas import ANIMAL_PROTEINS, EnrichedCatalogItem, parse_slm_json_output

DIETARY_TAGS_VOCAB = (
    "Zero Açúcar",
    "Sem Lactose",
    "Sem Glúten",
    "Vegano",
    "Vegetariano",
    "Low Carb/Fit",
)

SCALAR_FIELDS = (
    "category",
    "main_protein_or_base",
    "portion_size",
    "is_combo",
    "volume_or_weight",
)


def _tokenize_norm(items: list[str]) -> set[str]:
    tokens: set[str] = set()
    for text in items:
        words = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())
        for w in words:
            if len(w) >= 3:
                tokens.add(w)
    return tokens


def compute_token_overlap_f1(pred_list: list[str], gt_list: list[str]) -> float:
    """Calcula o F1 de sobreposição lexical entre listas de componentes ou sinônimos."""
    pred_tokens = _tokenize_norm(pred_list)
    gt_tokens = _tokenize_norm(gt_list)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    intersection = len(pred_tokens.intersection(gt_tokens))
    precision = intersection / len(pred_tokens)
    recall = intersection / len(gt_tokens)
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def compute_percentile(values: list[float], percentile: float) -> float:
    """Calcula percentil (ex: P50, P95) de forma determinística sem dependências pesadas."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(sorted_vals[int(k)], 3)
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return round(d0 + d1, 3)


def _check_raw_food_safety_hallucination(raw_output: str, gt_protein: str) -> bool:
    """Detecta se o modelo tentou emitir tag Vegano/Vegetariano para proteína animal real."""
    if gt_protein not in ANIMAL_PROTEINS:
        return False
    lowered = raw_output.lower()
    if '"dietary_tags"' not in lowered:
        return False
    match = re.search(r'"dietary_tags"\s*:\s*\[([^\]]*)\]', lowered)
    if not match:
        return False
    tags_segment = match.group(1)
    return ("vegano" in tags_segment) or ("vegetariano" in tags_segment)


def evaluate_predictions(
    raw_outputs: list[str],
    ground_truths: list[dict[str, Any]],
    latencies_ms: Optional[list[float]] = None,
) -> dict[str, Any]:
    """
    Avalia um lote de saídas brutas do SLM contra a lista de Ground Truths reais.
    Se uma saída falhar no JSON Schema, ela conta como erro (0) nas métricas de atributo
    para penalizar justamente modelos que quebram o contrato de produção.
    """
    if len(raw_outputs) != len(ground_truths):
        raise ValueError("raw_outputs e ground_truths devem ter exatamente o mesmo tamanho.")

    total = len(ground_truths)
    if total == 0:
        raise ValueError("A lista de avaliação não pode estar vazia.")

    valid_schema_count = 0
    strict_clean_json_count = 0
    food_safety_violations = 0

    field_hits = {field: 0 for field in SCALAR_FIELDS}
    components_f1_sum = 0.0
    synonyms_f1_sum = 0.0

    tag_tp = {tag: 0 for tag in DIETARY_TAGS_VOCAB}
    tag_fp = {tag: 0 for tag in DIETARY_TAGS_VOCAB}
    tag_fn = {tag: 0 for tag in DIETARY_TAGS_VOCAB}

    for raw_text, gt_dict in zip(raw_outputs, ground_truths):
        gt_item = EnrichedCatalogItem.model_validate(gt_dict)
        pred_item, err_msg, repaired = parse_slm_json_output(raw_text)

        if _check_raw_food_safety_hallucination(raw_text, gt_item.main_protein_or_base):
            food_safety_violations += 1
        elif err_msg and "Violação de segurança alimentar" in err_msg:
            food_safety_violations += 1

        gt_tags = set(gt_item.dietary_tags)

        if pred_item is None:
            for tag in gt_tags:
                tag_fn[tag] += 1
            continue

        valid_schema_count += 1
        if not repaired:
            strict_clean_json_count += 1

        if gt_item.main_protein_or_base in ANIMAL_PROTEINS:
            if {"Vegano", "Vegetariano"}.intersection(set(pred_item.dietary_tags)):
                food_safety_violations += 1

        for field in SCALAR_FIELDS:
            if getattr(pred_item, field) == getattr(gt_item, field):
                field_hits[field] += 1

        components_f1_sum += compute_token_overlap_f1(pred_item.components, gt_item.components)
        synonyms_f1_sum += compute_token_overlap_f1(pred_item.search_synonyms, gt_item.search_synonyms)

        pred_tags = set(pred_item.dietary_tags)
        for tag in DIETARY_TAGS_VOCAB:
            in_pred = tag in pred_tags
            in_gt = tag in gt_tags
            if in_pred and in_gt:
                tag_tp[tag] += 1
            elif in_pred and not in_gt:
                tag_fp[tag] += 1
            elif not in_pred and in_gt:
                tag_fn[tag] += 1

    total_tp = sum(tag_tp.values())
    total_fp = sum(tag_fp.values())
    total_fn = sum(tag_fn.values())

    micro_denom = (2 * total_tp) + total_fp + total_fn
    dietary_micro_f1 = (2 * total_tp / micro_denom) if micro_denom > 0 else 1.0

    per_tag_f1: dict[str, float] = {}
    for tag in DIETARY_TAGS_VOCAB:
        denom = (2 * tag_tp[tag]) + tag_fp[tag] + tag_fn[tag]
        per_tag_f1[tag] = round((2 * tag_tp[tag] / denom) if denom > 0 else 1.0, 4)

    dietary_macro_f1 = sum(per_tag_f1.values()) / len(DIETARY_TAGS_VOCAB)

    field_accuracies = {
        field: round(field_hits[field] / total, 4) for field in SCALAR_FIELDS
    }
    mean_field_exact_match = round(
        sum(field_accuracies.values()) / len(SCALAR_FIELDS), 4
    )

    latency_stats = {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    if latencies_ms:
        latency_stats = {
            "p50_ms": compute_percentile(latencies_ms, 50.0),
            "p95_ms": compute_percentile(latencies_ms, 95.0),
            "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
        }

    return {
        "total_samples": total,
        "json_schema_compliance_rate": round(valid_schema_count / total, 4),
        "strict_clean_json_rate": round(strict_clean_json_count / total, 4),
        "mean_field_exact_match": mean_field_exact_match,
        "field_accuracies": field_accuracies,
        "dietary_tags_micro_f1": round(dietary_micro_f1, 4),
        "dietary_tags_macro_f1": round(dietary_macro_f1, 4),
        "per_dietary_tag_f1": per_tag_f1,
        "components_token_f1": round(components_f1_sum / total, 4),
        "search_synonyms_token_f1": round(synonyms_f1_sum / total, 4),
        "food_safety_violations": food_safety_violations,
        "food_safety_violation_rate": round(food_safety_violations / total, 4),
        "latency": latency_stats,
    }


def simulate_unit_economics(
    monthly_skus: int = 50_000_000,
    frontier_input_tokens_per_sku: int = 950,
    frontier_output_tokens_per_sku: int = 220,
    frontier_input_cost_per_1m_usd: float = 2.50,
    frontier_output_cost_per_1m_usd: float = 10.00,
    slm_gpu_hourly_cost_usd: float = 0.75,
    slm_throughput_skus_per_sec: float = 32.0,
) -> dict[str, float]:
    """
    Modela o Unit Economics comparando:
    1) API LLM Proprietária (Few-Shot longo para forçar JSON Schema)
    2) SLM QLoRA Fine-Tuned Self-Hosted em GPU dedicada (vLLM / TGI com batching contínuo).
    """
    if monthly_skus <= 0 or slm_throughput_skus_per_sec <= 0:
        raise ValueError("monthly_skus e slm_throughput_skus_per_sec devem ser positivos.")

    frontier_in_cost_per_sku = (frontier_input_tokens_per_sku / 1_000_000.0) * frontier_input_cost_per_1m_usd
    frontier_out_cost_per_sku = (frontier_output_tokens_per_sku / 1_000_000.0) * frontier_output_cost_per_1m_usd
    frontier_total_monthly_usd = monthly_skus * (frontier_in_cost_per_sku + frontier_out_cost_per_sku)

    total_gpu_seconds = monthly_skus / slm_throughput_skus_per_sec
    total_gpu_hours = total_gpu_seconds / 3600.0
    slm_total_monthly_usd = total_gpu_hours * slm_gpu_hourly_cost_usd

    savings_monthly_usd = frontier_total_monthly_usd - slm_total_monthly_usd
    cost_reduction_pct = (savings_monthly_usd / frontier_total_monthly_usd) * 100.0

    return {
        "monthly_skus": float(monthly_skus),
        "frontier_llm_monthly_cost_usd": round(frontier_total_monthly_usd, 2),
        "slm_qlora_monthly_cost_usd": round(slm_total_monthly_usd, 2),
        "monthly_savings_usd": round(savings_monthly_usd, 2),
        "annualized_savings_usd": round(savings_monthly_usd * 12.0, 2),
        "cost_reduction_pct": round(cost_reduction_pct, 2),
        "frontier_cost_per_1k_skus_usd": round((frontier_total_monthly_usd / monthly_skus) * 1000.0, 4),
        "slm_cost_per_1k_skus_usd": round((slm_total_monthly_usd / monthly_skus) * 1000.0, 4),
        "slm_gpu_hours_required": round(total_gpu_hours, 2),
    }
