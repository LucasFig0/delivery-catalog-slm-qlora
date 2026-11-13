# iFood Catalog Intelligence: QLoRA Fine-Tuning of Qwen2.5-1.5B-Instruct

**Portfólio de Engenharia de IA — Projeto 2 (`ifood-catalog-slm-qlora`)**

Pipeline end-to-end de **Supervised Fine-Tuning (SFT) via QLoRA (4-bit NormalFloat)** sobre o Small Language Model **`Qwen/Qwen2.5-1.5B-Instruct`** para padronização de cardápios, extração estruturada de insumos (padrão ANVISA), classificação de segurança alimentar e expansão de busca semântica em português brasileiro.

---

## 1. Links Oficiais e Artefatos Publicados

- **Dataset no Hugging Face Hub (1.150 SKUs Reais EAN-13):** [`LucasFig0/brazilian-delivery-catalog-sft`](https://huggingface.co/datasets/LucasFig0/brazilian-delivery-catalog-sft)
- **Adaptador LoRA + Model Card no Hugging Face Hub:** [`LucasFig0/qwen2.5-1.5b-ifood-catalog-qlora`](https://huggingface.co/LucasFig0/qwen2.5-1.5b-ifood-catalog-qlora)
- **Notebook Reprodutível de Treino e Benchmark (GPU T4):** `notebooks/ifood_catalog_qlora_benchmark.ipynb`
- **Artigo Técnico Completo:** [`docs/TECHNICAL_ARTICLE.md`](docs/TECHNICAL_ARTICLE.md)
- **Métricas Auditadas (Holdout 200 SKUs):** [`artifacts/metrics/benchmark_results.json`](artifacts/metrics/benchmark_results.json)

---

## 2. Arquitetura do Sistema e Pipeline de Dados Reais (EAN-13)

Diferente de abordagens baseadas em dados sintéticos, este projeto opera sobre **1.150 SKUs reais** comercializados no Brasil extraídos do registro público **Open Food Facts Brasil** (`data/raw/openfoodfacts_br_raw.jsonl`), estratificados em 25 partições gastronômicas:

1. **Camada Bronze (`src/raw_ingestion.py`):** Ingestão resiliente via API paginada com deduplicação por código de barras **EAN-13 (`789...`)**, exigindo título real de PDV/ERP e declaração real de ingredientes em português.
2. **Camada Silver/Gold (`src/dataset_builder.py`):** Parser determinístico que isola insumos ignorando vírgulas aninhadas em colchetes/parênteses nutricionais, normaliza pesos/volumes (`350g`, `1.5L`, `500ml`), cruza tags de alergênicos (`en:gluten`, `en:milk`, `en:vegan`) e estrutura os pares em formato **ChatML (`system`, `user`, `assistant`)**.
3. **Contrato Estrito Pydantic v2 (`src/schemas.py`):** Validação com `ConfigDict(extra='forbid')` e **Guardrail Determinístico de Segurança Alimentar** (`@model_validator`) que bloqueia a atribuição de tags `Vegano` ou `Vegetariano` a itens com base proteica animal (`Bovino`, `Frango`, `Suíno`, `Peixe e Frutos do Mar`).
4. **Treinamento QLoRA (`src/qlora_config.py`):** Quantização NF4 4-bit com *Double Quantization* (`bitsandbytes`), adaptadores LoRA (`r=16, lora_alpha=32`) aplicados a todas as 7 projeções lineares do Qwen2.5 (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — **18.464.768 parâmetros treináveis**, ou **1,1820%** do total) e **Completion-Only Loss Masking** (`labels = -100` nos tokens de prompt).

---

## 3. Resultados Reais no Holdout Set (200 SKUs Inéditos — GPU NVIDIA T4)

Avaliação *Apples-to-Apples* comparando o modelo base `Qwen/Qwen2.5-1.5B-Instruct` (Zero-Shot com o mesmo `SYSTEM_PROMPT`) contra o modelo pós-SFT QLoRA nos 200 SKUs reais do split `test`:

| Métrica de Engenharia (Holdout n=200) | Base Zero-Shot | Fine-Tuned QLoRA | Delta | Diagnóstico de Produção |
| :--- | :---: | :---: | :---: | :--- |
| **JSON Schema Compliance Rate (%)** | 18.50% | **75.00%** | **+56.50 p.p.** | **4.05x** mais conformidade com contrato Pydantic v2 |
| **Strict Clean JSON (Sem Reparo) (%)** | 0.00% | **60.50%** | **+60.50 p.p.** | Emissão nativa sem cercas Markdown ou texto extra |
| **Mean Field Exact Match (%)** | 9.60% | **69.70%** | **+60.10 p.p.** | **7.26x** de ganho médio nos atributos escalares |
| **Category Accuracy (%)** | 5.50% | **75.00%** | **+69.50 p.p.** | **100.0% de acurácia condicional** nos JSONs válidos |
| **Main Protein/Base Accuracy (%)** | 3.00% | **61.50%** | **+58.50 p.p.** | **82.0% de acurácia condicional** nos JSONs válidos |
| **Portion Size Accuracy (%)** | 5.50% | **64.50%** | **+59.00 p.p.** | **86.0% de acurácia condicional** nos JSONs válidos |
| **Dietary Tags Micro-F1 (%)** | 10.22% | **31.84%** | **+21.62 p.p.** | **3.11x** de ganho em classificação multi-label |
| **Components Token F1 (%)** | 12.57% | **67.79%** | **+55.22 p.p.** | **90.4% de F1 condicional** sobre ingredientes ANVISA |
| **Search Synonyms Token F1 (%)** | 4.38% | **63.97%** | **+59.59 p.p.** | **85.3% de F1 condicional** para expansão de busca |
| **Food Safety Violations (Count)** | 11 (5.5%) | **0 (0.0%)** | **-11** | **100% de eliminação** de falsos positivos Vegano/Vegetariano |
| **Latency P50 (ms/SKU)** | 1123.61 ms | 1934.95 ms* | +811.34 ms | *Medido com 196 hooks PEFT dinâmicos não-fundidos |

### Nota Técnica sobre Latência e Fusão de Pesos (`merge_and_unload`)
Durante o benchmark no PyTorch/PEFT, o modelo avaliado executa 196 ganchos dinâmicos (`28 camadas x 7 projeções`) sobre pesos quantizados em NF4. Em produção (servido via **vLLM** ou **TGI**), a fusão linear dos adaptadores ($W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$) elimina 100% do overhead de dispatch do PEFT.

---

## 4. Unit Economics em Escala de Marketplace (50 Milhões de SKUs/mês)

| Parâmetro Financeiro / Operacional | API LLM Proprietária (Few-Shot) | SLM QLoRA Self-Hosted (GPU Dedicada) | Impacto / Economia |
| :--- | :---: | :---: | :---: |
| **Volume Mensal Processado** | 50.000.000 SKUs | 50.000.000 SKUs | Escala Nacional de Catálogo |
| **Custo por 1.000 SKUs Enriquecidos** | US$ 4.5750 | **US$ 0.0065** | **-99.86%** por lote |
| **Custo Mensal Total (COGS)** | US$ 228,750.00 | **US$ 325.52** | **US$ 228,424.48 / mês** |
| **Economia Anualizada Projetada** | — | — | **US$ 2,741,093.75 / ano** |

---

## 5. Como Executar Localmente e Rodar a Suíte de Testes (`pytest`)

```powershell
# 1. Ativar ambiente virtual e instalar dependências
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Reconstruir splits a partir do snapshot real (1.250 SKUs Open Food Facts BR)
python -m src.dataset_builder

# 3. Executar a suíte completa de testes automatizados
pytest -v
```