from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def generate_readme(metrics: dict) -> str:
    fence = chr(96) * 3
    econ = metrics["unit_economics_50m_skus_monthly"]

    lines = [
        "# iFood Catalog Intelligence: QLoRA Fine-Tuning of Qwen2.5-1.5B-Instruct",
        "",
        "**Portfólio de Engenharia de IA — Projeto 2 (`ifood-catalog-slm-qlora`)**",
        "",
        "Pipeline end-to-end de **Supervised Fine-Tuning (SFT) via QLoRA (4-bit NormalFloat)** sobre o Small Language Model **`Qwen/Qwen2.5-1.5B-Instruct`** para padronização de cardápios, extração estruturada de insumos (padrão ANVISA), classificação de segurança alimentar e expansão de busca semântica em português brasileiro.",
        "",
        "---",
        "",
        "## 1. Links Oficiais e Artefatos Publicados",
        "",
        "- **Dataset no Hugging Face Hub (1.150 SKUs Reais EAN-13):** [`LucasFig0/brazilian-delivery-catalog-sft`](https://huggingface.co/datasets/LucasFig0/brazilian-delivery-catalog-sft)",
        "- **Adaptador LoRA + Model Card no Hugging Face Hub:** [`LucasFig0/qwen2.5-1.5b-ifood-catalog-qlora`](https://huggingface.co/LucasFig0/qwen2.5-1.5b-ifood-catalog-qlora)",
        "- **Notebook Reprodutível de Treino e Benchmark (GPU T4):** `notebooks/ifood_catalog_qlora_benchmark.ipynb`",
        "- **Artigo Técnico Completo:** [`docs/TECHNICAL_ARTICLE.md`](docs/TECHNICAL_ARTICLE.md)",
        "- **Métricas Auditadas (Holdout 200 SKUs):** [`artifacts/metrics/benchmark_results.json`](artifacts/metrics/benchmark_results.json)",
        "",
        "---",
        "",
        "## 2. Arquitetura do Sistema e Pipeline de Dados Reais (EAN-13)",
        "",
        "Diferente de abordagens baseadas em dados sintéticos, este projeto opera sobre **1.150 SKUs reais** comercializados no Brasil extraídos do registro público **Open Food Facts Brasil** (`data/raw/openfoodfacts_br_raw.jsonl`), estratificados em 25 partições gastronômicas:",
        "",
        "1. **Camada Bronze (`src/raw_ingestion.py`):** Ingestão resiliente via API paginada com deduplicação por código de barras **EAN-13 (`789...`)**, exigindo título real de PDV/ERP e declaração real de ingredientes em português.",
        "2. **Camada Silver/Gold (`src/dataset_builder.py`):** Parser determinístico que isola insumos ignorando vírgulas aninhadas em colchetes/parênteses nutricionais, normaliza pesos/volumes (`350g`, `1.5L`, `500ml`), cruza tags de alergênicos (`en:gluten`, `en:milk`, `en:vegan`) e estrutura os pares em formato **ChatML (`system`, `user`, `assistant`)**.",
        "3. **Contrato Estrito Pydantic v2 (`src/schemas.py`):** Validação com `ConfigDict(extra='forbid')` e **Guardrail Determinístico de Segurança Alimentar** (`@model_validator`) que bloqueia a atribuição de tags `Vegano` ou `Vegetariano` a itens com base proteica animal (`Bovino`, `Frango`, `Suíno`, `Peixe e Frutos do Mar`).",
        "4. **Treinamento QLoRA (`src/qlora_config.py`):** Quantização NF4 4-bit com *Double Quantization* (`bitsandbytes`), adaptadores LoRA (`r=16, lora_alpha=32`) aplicados a todas as 7 projeções lineares do Qwen2.5 (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — **18.464.768 parâmetros treináveis**, ou **1,1820%** do total) e **Completion-Only Loss Masking** (`labels = -100` nos tokens de prompt).",
        "",
        "---",
        "",
        "## 3. Resultados Reais no Holdout Set (200 SKUs Inéditos — GPU NVIDIA T4)",
        "",
        "Avaliação *Apples-to-Apples* comparando o modelo base `Qwen/Qwen2.5-1.5B-Instruct` (Zero-Shot com o mesmo `SYSTEM_PROMPT`) contra o modelo pós-SFT QLoRA nos 200 SKUs reais do split `test`:",
        "",
        "| Métrica de Engenharia (Holdout n=200) | Base Zero-Shot | Fine-Tuned QLoRA | Delta | Diagnóstico de Produção |",
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
        "### Nota Técnica sobre Latência e Fusão de Pesos (`merge_and_unload`)",
        "Durante o benchmark no PyTorch/PEFT, o modelo avaliado executa 196 ganchos dinâmicos (`28 camadas x 7 projeções`) sobre pesos quantizados em NF4. Em produção (servido via **vLLM** ou **TGI**), a fusão linear dos adaptadores ($W_{\\text{merged}} = W_0 + \\frac{\\alpha}{r} B A$) elimina 100% do overhead de dispatch do PEFT.",
        "",
        "---",
        "",
        "## 4. Unit Economics em Escala de Marketplace (50 Milhões de SKUs/mês)",
        "",
        "| Parâmetro Financeiro / Operacional | API LLM Proprietária (Few-Shot) | SLM QLoRA Self-Hosted (GPU Dedicada) | Impacto / Economia |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Volume Mensal Processado** | 50.000.000 SKUs | 50.000.000 SKUs | Escala Nacional de Catálogo |",
        f"| **Custo por 1.000 SKUs Enriquecidos** | US$ {econ['frontier_cost_per_1k_skus_usd']:.4f} | **US$ {econ['slm_cost_per_1k_skus_usd']:.4f}** | **-{econ['cost_reduction_pct']:.2f}%** por lote |",
        f"| **Custo Mensal Total (COGS)** | US$ {econ['frontier_llm_monthly_cost_usd']:,.2f} | **US$ {econ['slm_qlora_monthly_cost_usd']:,.2f}** | **US$ {econ['monthly_savings_usd']:,.2f} / mês** |",
        f"| **Economia Anualizada Projetada** | — | — | **US$ {econ['annualized_savings_usd']:,.2f} / ano** |",
        "",
        "---",
        "",
        "## 5. Como Executar Localmente e Rodar a Suíte de Testes (`pytest`)",
        "",
        f"{fence}powershell",
        "# 1. Ativar ambiente virtual e instalar dependências",
        ".\\.venv\\Scripts\\Activate.ps1",
        "pip install -r requirements.txt",
        "",
        "# 2. Reconstruir splits a partir do snapshot real (1.250 SKUs Open Food Facts BR)",
        "python -m src.dataset_builder",
        "",
        "# 3. Executar a suíte completa de testes automatizados",
        "pytest -v",
        f"{fence}",
    ]
    return "\n".join(lines)


def generate_technical_article(metrics: dict) -> str:
    econ = metrics["unit_economics_50m_skus_monthly"]
    lines = [
        "# Fine-Tuning QLoRA de Small Language Models para Inteligência de Catálogo e Busca em Delivery: Do Código de Barras EAN-13 ao Unit Economics em Produção",
        "",
        "**Autor:** Lucas Figueiredo de Azevedo  ",
        "**Stack:** Python, PyTorch, Hugging Face (Transformers, PEFT, Datasets, bitsandbytes), Pydantic v2, Pytest, Docker.",
        "",
        "---",
        "",
        "## 1. Introdução: O Gargalo da Entropia Textual em Marketplaces de Delivery",
        "",
        "Em plataformas de delivery e supermercados online com dezenas de milhões de itens ativos, o catálogo é alimentado continuamente por integrações heterogêneas de ERPs, PDVs de restaurantes e planilhas manuais. O resultado é uma **alta entropia léxica**: títulos truncados como `ACHOC PO TODDY 370G` ou `AZEITE EXT VIRGEM ANDORINHA`, medidas sem padronização (`1,5 l`, `500 G`, `350ml`) e listas brutas de ingredientes em texto corrido segundo o padrão regulatório da ANVISA.",
        "",
        "Sem metadados estruturados — como categoria canônica, proteína principal, tamanho da porção, lista limpa de componentes, sinônimos semânticos e, sobretudo, **tags de restrição alimentar (`Sem Glúten`, `Sem Lactose`, `Zero Açúcar`, `Vegano`, `Vegetariano`)** — o motor de busca sofre queda de *Recall* e o usuário final fica exposto a riscos graves de segurança alimentar.",
        "",
        "Utilizar LLMs proprietários de fronteira (via API comercial) para enriquecer 50 milhões de SKUs mensalmente esbarra em dois obstáculos de engenharia:",
        "1. **Inviabilidade de Unit Economics (COGS):** Prompts *Few-Shot* extensos necessários para explicar regras de negócio consomem ~950 tokens de entrada e ~220 tokens de saída por SKU, gerando uma fatura superior a **US$ 232 mil por mês** (~US$ 2,78 milhões/ano).",
        "2. **Fragilidade de Contrato em Modelos Menores Zero-Shot:** Modelos compactos (1B a 3B de parâmetros) sem ajuste fino quebram frequentemente o esquema JSON ou alucinam tags incompatíveis com a biologia do alimento.",
        "",
        "Neste artigo, detalho a engenharia do projeto **`ifood-catalog-slm-qlora`**, no qual aplicamos **Supervised Fine-Tuning (SFT) com QLoRA (4-bit NormalFloat)** no modelo `Qwen/Qwen2.5-1.5B-Instruct` sobre **1.150 SKUs reais** do mercado brasileiro.",
        "",
        "---",
        "",
        "## 2. Engenharia de Dados: Curadoria de 1.150 SKUs Reais com Rastreabilidade EAN-13",
        "",
        "Um erro comum em projetos de demonstração de SFT é treinar o modelo sobre dados sintéticos gerados por regras ou por outro LLM. Modelos de 1,5 bilhão de parâmetros memorizam templates sintéticos em poucos steps, produzindo métricas artificialmente próximas de 100% que colapsam diante de dados reais.",
        "",
        "Para garantir validade ecológica e auditabilidade:",
        "- Extraímos um snapshot bruto de **1.250 produtos reais** comercializados no Brasil através da base pública **Open Food Facts Brasil**, estratificados em 25 partições de categorias (`pizzas`, `sandwiches`, `burgers`, `meals`, `meats`, `poultry`, `seafood`, `beverages`, `dairies`, `desserts`, `plant-based-foods`, entre outras).",
        "- Cada registro é identificado pelo seu código de barras oficial **`EAN-13`** (ex.: `EAN-7894900701517`), permitindo auditoria externa de qualquer amostra nos splits `train` (800), `validation` (150) e `test` (200), publicados em [`LucasFig0/brazilian-delivery-catalog-sft`](https://huggingface.co/datasets/LucasFig0/brazilian-delivery-catalog-sft).",
        "- O parser da Camada Silver analisa as declarações reais de ingredientes respeitando o aninhamento de colchetes e parênteses químicos (como `MINERAIS [CÁLCIO (CARBONATO DE CÁLCIO), FERRO E ZINCO]`), cruza rótulos nutricionais auditados (`labels_tags` e `allergens_tags`) e valida 100% dos registros contra o contrato `EnrichedCatalogItem` em **Pydantic v2**.",
        "",
        "---",
        "",
        "## 3. Arquitetura de Treinamento: QLoRA 4-bit + Completion-Only Loss Masking",
        "",
        "Para adaptar o `Qwen/Qwen2.5-1.5B-Instruct` dentro do envelope térmico e de memória de uma única **GPU NVIDIA Tesla T4 (16 GB VRAM)**, implementamos três decisões técnicas:",
        "",
        "1. **Quantização NormalFloat 4-bit (NF4) com Double Quantization:** Os pesos base congelados ($W_0$) são armazenados em 4 bits informacionalmente ótimos para distribuições normais, reduzindo a pegada dos pesos base para **~0,81 GB de VRAM**.",
        "2. **Adaptadores LoRA em Todas as Projeções Lineares (`r=16, alpha=32`):** Aplicamos matrizes de posto baixo $B A$ em todas as 7 camadas lineares de cada um dos 28 blocos Transformer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), totalizando exatamente **18.464.768 parâmetros treináveis (1,1820% do modelo)**.",
        "3. **Completion-Only Loss Masking:** Durante a tokenização das conversas ChatML, aplicamos a máscara `labels = -100` sobre todos os tokens de `system` e `user`. Dessa forma, o gradiente da *Cross-Entropy Loss* é retropropagado exclusivamente sobre os tokens do objeto JSON emitido pelo `assistant`, impedindo que o adaptador desperdice capacidade modelando o texto ruidoso de entrada.",
        "",
        "---",
        "",
        "## 4. Análise Experimental no Holdout Set (200 SKUs Reais Inéditos)",
        "",
        "O modelo foi avaliado em **200 SKUs reais inéditos** antes e após 2 épocas de treinamento QLoRA (100 steps de otimização com `paged_adamw_8bit` e scheduler cosseno):",
        "",
        "1. **Conformidade Estrita de Contrato (+56,50 p.p.):** O modelo Base Zero-Shot atingiu apenas **18,50%** de conformidade com o schema Pydantic v2 (e **0,00%** de JSON limpo sem cercas Markdown). Pós-QLoRA, o modelo saltou para **75,00%** de conformidade total e **60,50%** de JSON puro na primeira tentativa.",
        "2. **Acurácia Global vs. Acurácia Condicional:** Nosso avaliador penaliza rigorosamente qualquer quebra de schema atribuindo nota zero a todos os campos daquela amostra. Analisando a performance condicional dentro dos 75% de respostas que passaram pelo validador Pydantic v2, o modelo QLoRA atingiu:",
        "   - **100,0% de acurácia na classificação de Categoria Canônica** (75,00% global / 75,00% válidos);",
        "   - **90,39% de Token F1 na extração de componentes/ingredientes reais** (67,79% global);",
        "   - **86,00% de acurácia em Portion Size** (64,50% global);",
        "   - **85,29% de Token F1 na geração de sinônimos de busca** (63,97% global).",
        "3. **Segurança Alimentar (Zero Violações):** Enquanto o modelo Base Zero-Shot cometeu **11 violações graves** (rotulando produtos cárneos como `Vegano` ou `Vegetariano`), o modelo pós-QLoRA combinado ao guardrail determinístico reduziu as violações para **0 (0,00%)**.",
        "",
        "---",
        "",
        "## 5. Conclusão e Viabilidade Econômica em Produção",
        "",
        f"Ao substituir chamadas a LLMs proprietários de grande porte por um SLM de 1,5B ajustado com QLoRA e servido em infraestrutura dedicada com fusão de pesos (`merge_and_unload`), o custo por lote de 1.000 SKUs cai de **US$ {econ['frontier_cost_per_1k_skus_usd']:.4f}** para **US$ {econ['slm_cost_per_1k_skus_usd']:.4f}** — uma redução de **{econ['cost_reduction_pct']:.2f}%** no COGS, representando uma economia anual projetada de **US$ {econ['annualized_savings_usd']:,.2f}** para uma operação de 50 milhões de itens mensais.",
    ]
    return "\n".join(lines)


def main() -> None:
    metrics_path = ROOT_DIR / "artifacts" / "metrics" / "benchmark_results.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    readme_path = ROOT_DIR / "README.md"
    readme_path.write_text(generate_readme(metrics), encoding="utf-8")
    print(f"README.md gerado com sucesso em: {readme_path}")

    article_path = ROOT_DIR / "docs" / "TECHNICAL_ARTICLE.md"
    article_path.parent.mkdir(parents=True, exist_ok=True)
    article_path.write_text(generate_technical_article(metrics), encoding="utf-8")
    print(f"Artigo Técnico gerado com sucesso em: {article_path}")


if __name__ == "__main__":
    main()
