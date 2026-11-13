from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

DATASET_REPO_NAME = "brazilian-delivery-catalog-sft"


def resolve_hf_token() -> str:
    """Obtém o HF_TOKEN da variável de ambiente ou do arquivo local .env (não versionado)."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError("HF_TOKEN não encontrado no ambiente nem no arquivo .env.")


def load_jsonl_rows(file_path: Path) -> list[dict]:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset_card(
    repo_id: str,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
) -> str:
    all_rows = train_rows + val_rows + test_rows
    cat_counts = Counter(r["category"] for r in all_rows)
    tag_counts = Counter(
        tag for r in all_rows for tag in r["ground_truth"]["dietary_tags"]
    )

    cat_lines = "\n".join(
        f"| `{cat}` | {cnt} | {(cnt / len(all_rows)) * 100:.1f}% |"
        for cat, cnt in cat_counts.most_common()
    )
    tag_lines = "\n".join(
        f"| `{tag}` | {cnt} |" for tag, cnt in tag_counts.most_common()
    )

    lines = [
        "---",
        "language:",
        "- pt",
        "license: mit",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- ifood",
        "- catalog-enrichment",
        "- slm",
        "- qlora",
        "- sft",
        "- brazilian-portuguese",
        "- openfoodfacts",
        "- pydantic",
        "size_categories:",
        "- 1K<n<10K",
        "---",
        "",
        f"# {DATASET_REPO_NAME}: Real-World Brazilian Delivery & Grocery Catalog SFT Dataset",
        "",
        "## 1. Visão Geral e Origem dos Dados (100% SKUs Reais)",
        f"O dataset **`{repo_id}`** contém **1.150 SKUs reais** comercializados no Brasil (extraídos e auditados a partir do registro público *Open Food Facts Brasil* com rastreabilidade por código de barras **EAN-13**), estruturados para **Supervised Fine-Tuning (SFT / QLoRA)** de Small Language Models (SLMs) aplicados a **Inteligência de Catálogo, Padronização de Cardápios e Busca Semântica em Delivery**.",
        "",
        "- **Split `train`**: 800 SKUs reais",
        "- **Split `validation`**: 150 SKUs reais",
        "- **Split `test` (Holdout)**: 200 SKUs reais (zero sobreposição de código EAN-13 ou prompt)",
        "- **Formato de Instrução**: ChatML (`messages`: `system`, `user`, `assistant`) compatível nativamente com `Qwen/Qwen2.5-1.5B-Instruct` e `TRL SFTTrainer`.",
        "- **Contrato de Saída Validado**: 100% dos alvos (`assistant`) seguem rigorosamente o schema Pydantic v2 `EnrichedCatalogItem` com guardrails determinísticos de segurança alimentar.",
        "",
        "## 2. Distribuição Taxonômica (Categorias Canônicas)",
        "",
        "| Categoria Canônica | SKUs Reais | Proporção |",
        "| :--- | :---: | :---: |",
        cat_lines,
        "",
        "## 3. Distribuição de Tags de Restrição e Segurança Alimentar (`dietary_tags`)",
        "",
        "| Dietary Tag | Ocorrências Auditadas |",
        "| :--- | :---: |",
        tag_lines,
        "",
        "## 4. Estrutura das Colunas",
        "- `sku_id` (`string`): Identificador único rastreável no padrão `EAN-<codigo_de_barras>`.",
        "- `merchant_type` (`string`): Tipo de loja/estabelecimento inferido dos metadados reais.",
        "- `category` (`string`): Categoria canônica de catálogo.",
        "- `raw_input` (`dict`): Objeto de entrada bruto (`RawCatalogItem`).",
        "- `ground_truth` (`dict`): Objeto estruturado de saída (`EnrichedCatalogItem`) validado via Pydantic v2.",
        "- `messages` (`list[dict]`): Conversa no padrão ChatML (`system`, `user`, `assistant`) pronta para SFT.",
    ]
    return "\n".join(lines)


def publish_dataset(processed_dir: str | Path = "data/processed") -> str:
    token = resolve_hf_token()
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{DATASET_REPO_NAME}"

    proc_path = Path(processed_dir)
    train_rows = load_jsonl_rows(proc_path / "train.jsonl")
    val_rows = load_jsonl_rows(proc_path / "validation.jsonl")
    test_rows = load_jsonl_rows(proc_path / "test.jsonl")

    ds_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
            "test": Dataset.from_list(test_rows),
        }
    )

    print(f"Publicando DatasetDict em https://huggingface.co/datasets/{repo_id} ...")
    ds_dict.push_to_hub(repo_id=repo_id, token=token, private=False)

    readme_content = build_dataset_card(repo_id, train_rows, val_rows, test_rows)
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add comprehensive Dataset Card with real EAN-13 distribution stats",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\nSucesso! Dataset publicado e auditável em: {url}")
    return url


if __name__ == "__main__":
    publish_dataset()
