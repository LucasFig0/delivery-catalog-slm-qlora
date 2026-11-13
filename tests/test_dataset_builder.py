import json
from pathlib import Path

from src.dataset_builder import build_sft_dataset, extract_clean_measure, extract_real_components, save_splits_to_jsonl
from src.schemas import parse_slm_json_output


def test_extract_clean_measure_real_examples():
    assert extract_clean_measure("350 g", "Achocolatado Nescau") == "350g"
    assert extract_clean_measure("2 l", "Coca Cola Zero") == "2L"
    assert extract_clean_measure("", "Leite UHT 1 Litro") == "1L"


def test_extract_real_components_ignores_nested_commas():
    raw_ing = (
        "INGREDIENTES: AÇÚCAR, CACAU EM PÓ, MINERAIS [CÁLCIO (CARBONATO DE CÁLCIO), FERRO E ZINCO], "
        "MALTODEXTRINA, EMULSIFICANTE LECITINA DE SOJA. ALÉRGICOS: CONTÉM DERIVADOS DE SOJA."
    )
    comps = extract_real_components(raw_ing, "Achocolatado")
    assert len(comps) >= 3
    assert any("Açúcar" in c or "açúcar" in c.lower() for c in comps)
    assert all("alérgicos" not in c.lower() for c in comps)


def test_real_sft_dataset_zero_leakage_and_ean_traceability():
    splits = build_sft_dataset(train_size=800, val_size=150, test_size=200, seed=42)
    assert len(splits["train"]) == 800
    assert len(splits["validation"]) == 150
    assert len(splits["test"]) == 200

    all_records = splits["train"] + splits["validation"] + splits["test"]
    for rec in all_records:
        assert rec["sku_id"].startswith("EAN-")
        parsed, err, repaired = parse_slm_json_output(rec["messages"][2]["content"])
        assert err is None
        assert repaired is False
        assert parsed is not None

    train_skus = {r["sku_id"] for r in splits["train"]}
    val_skus = {r["sku_id"] for r in splits["validation"]}
    test_skus = {r["sku_id"] for r in splits["test"]}

    assert train_skus.isdisjoint(val_skus)
    assert train_skus.isdisjoint(test_skus)
    assert val_skus.isdisjoint(test_skus)


def test_save_real_splits_to_jsonl(tmp_path: Path):
    splits = build_sft_dataset(train_size=20, val_size=5, test_size=5, seed=7)
    saved = save_splits_to_jsonl(splits, output_dir=tmp_path)
    assert saved["train"].exists()
    first = json.loads(saved["train"].read_text(encoding="utf-8").splitlines()[0])
    assert first["sku_id"].startswith("EAN-")
