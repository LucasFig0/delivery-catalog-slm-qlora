import json

from src.evaluator import (
    compute_token_overlap_f1,
    evaluate_predictions,
    simulate_unit_economics,
)


def _sample_ground_truth() -> dict:
    return {
        "normalized_title": "Filé de Frango à Parmegiana com Arroz e Fritas 500g",
        "category": "Brasileira",
        "main_protein_or_base": "Frango",
        "portion_size": "Individual",
        "dietary_tags": ["Sem Lactose"],
        "is_combo": False,
        "volume_or_weight": "500g",
        "components": ["Filé de frango", "Molho de tomate", "Arroz branco", "Batata frita"],
        "search_synonyms": ["parmegiana de frango", "prato executivo", "almoco"],
    }


def test_evaluate_perfect_predictions():
    gt = _sample_ground_truth()
    raw_perfect = json.dumps(gt, ensure_ascii=False)
    report = evaluate_predictions(
        raw_outputs=[raw_perfect, raw_perfect],
        ground_truths=[gt, gt],
        latencies_ms=[110.0, 130.0],
    )
    assert report["json_schema_compliance_rate"] == 1.0
    assert report["strict_clean_json_rate"] == 1.0
    assert report["mean_field_exact_match"] == 1.0
    assert report["dietary_tags_micro_f1"] == 1.0
    assert report["food_safety_violations"] == 0
    assert report["latency"]["p50_ms"] == 120.0


def test_evaluate_penalizes_broken_json_and_food_safety_violation():
    gt = _sample_ground_truth()
    raw_valid_fenced = f"{chr(96)*3}json\n{json.dumps(gt, ensure_ascii=False)}\n{chr(96)*3}"

    bad_vegan_chicken = dict(gt)
    bad_vegan_chicken["dietary_tags"] = ["Vegano"]
    raw_unsafe = json.dumps(bad_vegan_chicken, ensure_ascii=False)

    report = evaluate_predictions(
        raw_outputs=[raw_valid_fenced, raw_unsafe],
        ground_truths=[gt, gt],
    )
    assert report["json_schema_compliance_rate"] == 0.5
    assert report["strict_clean_json_rate"] == 0.0
    assert report["food_safety_violations"] == 1
    assert report["food_safety_violation_rate"] == 0.5


def test_token_overlap_f1():
    score = compute_token_overlap_f1(
        ["Filé de frango grelhado", "Arroz branco"],
        ["Filé de frango", "Arroz branco soltinho"],
    )
    assert 0.65 <= score <= 1.0


def test_simulate_unit_economics_50m_skus():
    econ = simulate_unit_economics(monthly_skus=50_000_000)
    assert econ["frontier_llm_monthly_cost_usd"] > 100_000.0
    assert econ["slm_qlora_monthly_cost_usd"] < 1_000.0
    assert econ["cost_reduction_pct"] > 99.0
    assert econ["annualized_savings_usd"] > 1_000_000.0
