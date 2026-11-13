# Fine-Tuning QLoRA de Small Language Models para Inteligência de Catálogo e Busca em Delivery: Do Código de Barras EAN-13 ao Unit Economics em Produção

**Autor:** Lucas Figueiredo de Azevedo  
**Stack:** Python, PyTorch, Hugging Face (Transformers, PEFT, Datasets, bitsandbytes), Pydantic v2, Pytest, Docker.

---

## 1. Introdução: O Gargalo da Entropia Textual em Marketplaces de Delivery

Em plataformas de delivery e supermercados online com dezenas de milhões de itens ativos, o catálogo é alimentado continuamente por integrações heterogêneas de ERPs, PDVs de restaurantes e planilhas manuais. O resultado é uma **alta entropia léxica**: títulos truncados como `ACHOC PO TODDY 370G` ou `AZEITE EXT VIRGEM ANDORINHA`, medidas sem padronização (`1,5 l`, `500 G`, `350ml`) e listas brutas de ingredientes em texto corrido segundo o padrão regulatório da ANVISA.

Sem metadados estruturados — como categoria canônica, proteína principal, tamanho da porção, lista limpa de componentes, sinônimos semânticos e, sobretudo, **tags de restrição alimentar (`Sem Glúten`, `Sem Lactose`, `Zero Açúcar`, `Vegano`, `Vegetariano`)** — o motor de busca sofre queda de *Recall* e o usuário final fica exposto a riscos graves de segurança alimentar.

Utilizar LLMs proprietários de fronteira (via API comercial) para enriquecer 50 milhões de SKUs mensalmente esbarra em dois obstáculos de engenharia:
1. **Inviabilidade de Unit Economics (COGS):** Prompts *Few-Shot* extensos necessários para explicar regras de negócio consomem ~950 tokens de entrada e ~220 tokens de saída por SKU, gerando uma fatura superior a **US$ 232 mil por mês** (~US$ 2,78 milhões/ano).
2. **Fragilidade de Contrato em Modelos Menores Zero-Shot:** Modelos compactos (1B a 3B de parâmetros) sem ajuste fino quebram frequentemente o esquema JSON ou alucinam tags incompatíveis com a biologia do alimento.

Neste artigo, detalho a engenharia do projeto **`ifood-catalog-slm-qlora`**, no qual aplicamos **Supervised Fine-Tuning (SFT) com QLoRA (4-bit NormalFloat)** no modelo `Qwen/Qwen2.5-1.5B-Instruct` sobre **1.150 SKUs reais** do mercado brasileiro.

---

## 2. Engenharia de Dados: Curadoria de 1.150 SKUs Reais com Rastreabilidade EAN-13

Um erro comum em projetos de demonstração de SFT é treinar o modelo sobre dados sintéticos gerados por regras ou por outro LLM. Modelos de 1,5 bilhão de parâmetros memorizam templates sintéticos em poucos steps, produzindo métricas artificialmente próximas de 100% que colapsam diante de dados reais.

Para garantir validade ecológica e auditabilidade:
- Extraímos um snapshot bruto de **1.250 produtos reais** comercializados no Brasil através da base pública **Open Food Facts Brasil**, estratificados em 25 partições de categorias (`pizzas`, `sandwiches`, `burgers`, `meals`, `meats`, `poultry`, `seafood`, `beverages`, `dairies`, `desserts`, `plant-based-foods`, entre outras).
- Cada registro é identificado pelo seu código de barras oficial **`EAN-13`** (ex.: `EAN-7894900701517`), permitindo auditoria externa de qualquer amostra nos splits `train` (800), `validation` (150) e `test` (200), publicados em [`LucasFig0/brazilian-delivery-catalog-sft`](https://huggingface.co/datasets/LucasFig0/brazilian-delivery-catalog-sft).
- O parser da Camada Silver analisa as declarações reais de ingredientes respeitando o aninhamento de colchetes e parênteses químicos (como `MINERAIS [CÁLCIO (CARBONATO DE CÁLCIO), FERRO E ZINCO]`), cruza rótulos nutricionais auditados (`labels_tags` e `allergens_tags`) e valida 100% dos registros contra o contrato `EnrichedCatalogItem` em **Pydantic v2**.

---

## 3. Arquitetura de Treinamento: QLoRA 4-bit + Completion-Only Loss Masking

Para adaptar o `Qwen/Qwen2.5-1.5B-Instruct` dentro do envelope térmico e de memória de uma única **GPU NVIDIA Tesla T4 (16 GB VRAM)**, implementamos três decisões técnicas:

1. **Quantização NormalFloat 4-bit (NF4) com Double Quantization:** Os pesos base congelados ($W_0$) são armazenados em 4 bits informacionalmente ótimos para distribuições normais, reduzindo a pegada dos pesos base para **~0,81 GB de VRAM**.
2. **Adaptadores LoRA em Todas as Projeções Lineares (`r=16, alpha=32`):** Aplicamos matrizes de posto baixo $B A$ em todas as 7 camadas lineares de cada um dos 28 blocos Transformer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), totalizando exatamente **18.464.768 parâmetros treináveis (1,1820% do modelo)**.
3. **Completion-Only Loss Masking:** Durante a tokenização das conversas ChatML, aplicamos a máscara `labels = -100` sobre todos os tokens de `system` e `user`. Dessa forma, o gradiente da *Cross-Entropy Loss* é retropropagado exclusivamente sobre os tokens do objeto JSON emitido pelo `assistant`, impedindo que o adaptador desperdice capacidade modelando o texto ruidoso de entrada.

---

## 4. Análise Experimental no Holdout Set (200 SKUs Reais Inéditos)

O modelo foi avaliado em **200 SKUs reais inéditos** antes e após 2 épocas de treinamento QLoRA (100 steps de otimização com `paged_adamw_8bit` e scheduler cosseno):

1. **Conformidade Estrita de Contrato (+56,50 p.p.):** O modelo Base Zero-Shot atingiu apenas **18,50%** de conformidade com o schema Pydantic v2 (e **0,00%** de JSON limpo sem cercas Markdown). Pós-QLoRA, o modelo saltou para **75,00%** de conformidade total e **60,50%** de JSON puro na primeira tentativa.
2. **Acurácia Global vs. Acurácia Condicional:** Nosso avaliador penaliza rigorosamente qualquer quebra de schema atribuindo nota zero a todos os campos daquela amostra. Analisando a performance condicional dentro dos 75% de respostas que passaram pelo validador Pydantic v2, o modelo QLoRA atingiu:
   - **100,0% de acurácia na classificação de Categoria Canônica** (75,00% global / 75,00% válidos);
   - **90,39% de Token F1 na extração de componentes/ingredientes reais** (67,79% global);
   - **86,00% de acurácia em Portion Size** (64,50% global);
   - **85,29% de Token F1 na geração de sinônimos de busca** (63,97% global).
3. **Segurança Alimentar (Zero Violações):** Enquanto o modelo Base Zero-Shot cometeu **11 violações graves** (rotulando produtos cárneos como `Vegano` ou `Vegetariano`), o modelo pós-QLoRA combinado ao guardrail determinístico reduziu as violações para **0 (0,00%)**.

---

## 5. Conclusão e Viabilidade Econômica em Produção

Ao substituir chamadas a LLMs proprietários de grande porte por um SLM de 1,5B ajustado com QLoRA e servido em infraestrutura dedicada com fusão de pesos (`merge_and_unload`), o custo por lote de 1.000 SKUs cai de **US$ 4.5750** para **US$ 0.0065** — uma redução de **99.86%** no COGS, representando uma economia anual projetada de **US$ 2,741,093.75** para uma operação de 50 milhões de itens mensais.