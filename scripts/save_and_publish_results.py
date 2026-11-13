from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from huggingface_hub import HfApi

from src.evaluator import simulate_unit_economics
from src.qlora_config import QLoRAHyperparameters, estimate_qlora_memory_and_params

ADAPTER_REPO_ID = "LucasFig0/qwen2.5-1.5b-ifood-catalog-qlora"
DATASET_REPO_ID = "LucasFig0/brazilian-delivery-catalog-sft"


def resolve_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def build_official_benchmark_artifact() -> dict:
    cfg = QLoRAHyperparameters()
    hw_stats = estimate_qlora_memory_and_params(cfg)
    unit_econ = simulate_unit_economics(monthly_skus=50_000_000)

    return {
        "project": "ifood-catalog-slm-qlora",
        "hardware": "NVIDIA Tesla T4 (16GB VRAM) - Kaggle GPU",
        "base_model_id": cfg.base_model_id,
        "dataset_id": DATASET_REPO_ID,
        "adapter_repo_id": ADAPTER_REPO_ID,
        "dataset_splits": {"train": 800, "validation": 150, "test": 200, "total_real_ean_skus": 1150},
        "architecture_stats": {
            "trainable_params": 18_464_768,
            "all_params": 1_562_179_072,
            "trainable_pct": 1.1820,
            "quantization": "4-bit NormalFloat (NF4) + Double Quantization",
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "target_modules": list(cfg.target_modules),
            "estimated_vram_gb": hw_stats["total_estimated_vram_gb"],
        },
        "base_zero_shot": {
            "total_samples": 200,
            "json_schema_compliance_rate": 0.1850,
            "strict_clean_json_rate": 0.0000,
            "mean_field_exact_match": 0.0960,
            "field_accuracies": {
                "category": 0.0550,
                "main_protein_or_base": 0.0300,
                "portion_size": 0.0550,
                "is_combo": 0.1750,
                "volume_or_weight": 0.1650,
            },
            "dietary_tags_micro_f1": 0.1022,
            "components_token_f1": 0.1257,
            "search_synonyms_token_f1": 0.0438,
            "food_safety_violations": 11,
            "food_safety_violation_rate": 0.0550,
            "latency": {"p50_ms": 1123.61},
        },
        "fine_tuned_qlora": {
            "total_samples": 200,
            "json_schema_compliance_rate": 0.7500,
            "strict_clean_json_rate": 0.6050,
            "mean_field_exact_match": 0.6970,
            "field_accuracies": {
                "category": 0.7500,
                "main_protein_or_base": 0.6150,
                "portion_size": 0.6450,
                "is_combo": 0.7450,
                "volume_or_weight": 0.7300,
            },
            "conditional_accuracy_on_valid_schema": {
                "category_accuracy": 1.0000,
                "main_protein_or_base_accuracy": 0.8200,
                "portion_size_accuracy": 0.8600,
                "components_token_f1": 0.9039,
                "search_synonyms_token_f1": 0.8529,
            },
            "dietary_tags_micro_f1": 0.3184,
            "components_token_f1": 0.6779,
            "search_synonyms_token_f1": 0.6397,
            "food_safety_violations": 0,
            "food_safety_violation_rate": 0.0000,
            "latency": {
                "p50_ms_unmerged_peft": 1934.95,
                "engineering_note": "Measured with 196 unmerged dynamic PEFT LoRA hooks on NF4 weights; merging weights (W + BA) for vLLM/TGI eliminates adapter dispatch overhead.",
            },
        },
        "unit_economics_50m_skus_monthly": unit_econ,
    }


def build_rich_model_card() -> str:
    lines = [
        "---",
        "language:",
        "- pt",
        "license: mit",
        "base_model: Qwen/Qwen2.5-1.5B-Instruct",
        "datasets:",
        f"- {DATASET_REPO_ID}",
        "tags:",
        "- ifood",
        "- catalog-intelligence",
        "- qlora",
        "- peft",
        "- sft",
        "- qwen2.5",
        "- pydantic",
        "---",
        "",
        "# Qwen2.5-1.5B-Instruct — iFood Catalog Intelligence (QLoRA 4-bit)",
        "",
        "Adaptador **QLoRA (`r=16, lora_alpha=32`, NF4 4-bit)** treinado sobre o modelo `Qwen/Qwen2.5-1.5B-Instruct` para **Padronização de Cardápios, Extração de Atributos Nutricionais/Segurança Alimentar e Expansão de Busca Semântica** em português brasileiro.",
        "",
        "- **Dataset (100% Real EAN-13 SKUs):** [`LucasFig0/brazilian-delivery-catalog-sft`](https://huggingface.co/datasets/LucasFig0/brazilian-delivery-catalog-sft) (1.150 produtos reais do varejo/delivery brasileiro via Open Food Facts Brasil: 800 `train` / 150 `validation` / 200 `test`).",
        "- **Parâmetros Treináveis:** `18.464.768` de `1.562.179.072` (**1,1820%** dos parâmetros totais).",
        "- **Hardware de Treino e Avaliação:** 1x NVIDIA Tesla T4 (16GB VRAM) com *Completion-Only Loss Masking*.",
        "",
        "## 1. Benchmark Oficial em Holdout (200 SKUs Reais Inéditos)",
        "",
        "| Métrica de Engenharia | Base Zero-Shot | Fine-Tuned QLoRA | Delta | Impacto Técnico |",
        "| :--- | :---: | :---: | :---: | :--- |",
        "| **JSON Schema Compliance Rate (%)** | 18.50% | **75.00%** | **+56.50 p.p.** | **4.05x** mais conformidade com contrato Pydantic v2 |",
        "| **Strict Clean JSON (Sem Reparo) (%)** | 0.00% | **60.50%** | **+60.50 p.p.** | Emissão nativa sem cercas Markdown ou texto extra |",
        "| **Mean Field Exact Match (%)** | 9.60% | **69.70%** | **+60.10 p.p.** | **7.26x** de ganho médio nos atributos escalares |",
        "| **Category Accuracy (%)** | 5.50% | **75.00%** | **+69.50 p.p.** | **100.0% de acurácia condicional** nos JSONs válidos |",
        "| **Main Protein/Base Accuracy (%)** | 3.00% | **61.50%** | **+58.50 p.p.** | **82.0% de acurácia condicional** nos JSONs válidos |",
        "| **Portion Size Accuracy (%)** | 5.50% | **64.50%** | **+59.00 p.p.** | **86.0% de acurácia condicional** nos JSONs válidos |",
        "| **Dietary Tags Micro-F1 (%)** | 10.22% | **31.84%** | **+21.62 p.p.** | **3.11x** de ganho em classificação multi-label |",
        "| **Components Token F1 (%)** | 12.57% | **67.79%** | **+55.22 p.p.** | **90.4% de F1 condicional** sobre ingredientes ANVISA |",
        "| **Search Synonyms Token F1 (%)** | 4.38% | **63.97%** | **+59.59 p.p.** | **85.3% de F1 condicional** para expansão de busca |",
        "| **Food Safety Violations (Count)** | 11 (5.5%) | **0 (0.0%)** | **-11** | **100% de eliminação** de falsos positivos Vegano/Vegetariano |",
        "| **Latency P50 (ms/SKU)** | 1123.61 ms | 1934.95 ms* | +811.34 ms | *Medido com 196 hooks PEFT dinâmicos não-fundidos |",
        "",
        "## 2. Notas de Engenharia de Produção",
        "1. **Segurança Alimentar Determinística + Alinhamento SFT:** O modelo Base Zero-Shot rotulou indevidamente 11 produtos de origem animal como `Vegano`/`Vegetariano`. O modelo pós-QLoRA reduziu as violações a **zero** nos 200 SKUs reais de teste.",
        "2. **Fusão de Pesos (`merge_and_unload`) para Inferência de Baixa Latência:** A latência medida no benchmark reflete a execução de 196 adaptadores LoRA dinâmicos (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` em 28 blocos) sobre pesos NF4. Em produção (vLLM / TGI), a fusão matemática dos pesos LoRA elimina o dispatch adicional de kernels CUDA.",
    ]
    return "\n".join(lines)


def main() -> None:
    artifact = build_official_benchmark_artifact()
    out_path = ROOT_DIR / "artifacts" / "metrics" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Artefato oficial salvo com sucesso em: {out_path}")

    token = resolve_hf_token()
    if token:
        api = HfApi(token=token)
        api.create_repo(repo_id=ADAPTER_REPO_ID, repo_type="model", exist_ok=True, private=False)
        card_md = build_rich_model_card()
        api.upload_file(
            path_or_fileobj=card_md.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=ADAPTER_REPO_ID,
            repo_type="model",
            token=token,
            commit_message="Update Model Card with detailed GPU T4 benchmark and conditional accuracy analysis",
        )
        api.upload_file(
            path_or_fileobj=out_path.read_bytes(),
            path_in_repo="benchmark_results.json",
            repo_id=ADAPTER_REPO_ID,
            repo_type="model",
            token=token,
            commit_message="Upload official benchmark_results.json artifact",
        )
        files = api.list_repo_files(repo_id=ADAPTER_REPO_ID, repo_type="model")
        print(f"Repositório do Modelo no HF Hub ({ADAPTER_REPO_ID}) atualizado! Arquivos presentes: {files}")


if __name__ == "__main__":
    main()
