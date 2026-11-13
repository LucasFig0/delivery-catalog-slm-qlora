import json
from pathlib import Path


def test_official_benchmark_results_artifact_exists_and_meets_quality_gates():
    artifact_path = Path("artifacts/metrics/benchmark_results.json")
    assert artifact_path.exists(), "O arquivo artifacts/metrics/benchmark_results.json deve existir."

    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    base = data["base_zero_shot"]
    ft = data["fine_tuned_qlora"]
    econ = data["unit_economics_50m_skus_monthly"]

    # Gate 1: Salto expressivo em JSON Schema Compliance (18.5% -> 75.0%)
    assert ft["json_schema_compliance_rate"] >= 0.70
    assert ft["json_schema_compliance_rate"] > base["json_schema_compliance_rate"] * 3.5

    # Gate 2: Eliminação total de violações de segurança alimentar (11 -> 0)
    assert base["food_safety_violations"] == 11
    assert ft["food_safety_violations"] == 0
    assert ft["food_safety_violation_rate"] == 0.0

    # Gate 3: Ganho consistente de F1 em extração de componentes e sinônimos de busca
    assert ft["components_token_f1"] > base["components_token_f1"] * 4.5
    assert ft["search_synonyms_token_f1"] > base["search_synonyms_token_f1"] * 10.0

    # Gate 4: Unit Economics viável em escala de 50M SKUs/mês
    assert econ["cost_reduction_pct"] > 99.0
