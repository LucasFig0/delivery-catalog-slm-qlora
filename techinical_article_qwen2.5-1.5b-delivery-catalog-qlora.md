Fine-Tuning QLoRA de Small Language Models para
Inteligência de Catálogo e Busca em Delivery
Do Código de Barras EAN-13 ao Unit Economics em Produção
Autor: Lucas Figueiredo de Azevedo
Projeto pessoal e independente, sem vínculo com nenhuma empresa de delivery ou marketplace.
Resumo Executivo
Enriquecer o catálogo de um marketplace de delivery (títulos truncados de nota fiscal, unidades
sem padrão, ingredientes em texto corrido) com LLMs proprietários via API custa, no modelo de
custos deste trabalho, US$ 228.750,00/mês para 50 milhões de SKUs. Small Language Models em
Zero-Shot são baratos, mas quebram o esquema JSON e cometem erros de segurança alimentar,
como marcar como Vegano um produto de origem animal.
O projeto delivery-catalog-slm-qlora aplica SFT via QLoRA (4-bit NF4) ao Qwen2.5-1.5B-
Instruct sobre 1.150 SKUs reais do Open Food Facts Brasil, identificados por EAN-13 e
particionados sem data leakage (800 treino, 150 validação, 200 teste). Um schema Pydantic v2
valida toda saída e bloqueia combinações inválidas de proteína animal com tags
Vegano/Vegetariano.
Resultados no holdout (200 SKUs inéditos, GPU T4):
Conformidade com o schema JSON: de 18,50% para 75,00% (4,05x)
Category Accuracy: de 5,50% para 75,00% no global e 100% condicional aos JSONs válidos
Violações de segurança alimentar: de 11 (5,5%) para 0 no holdout
Dietary Tags Micro-F1: de 10,22% para 31,84%, ainda baixo (ver Limitações)
Custo mensal modelado: de US$ 228.750,00 para US$ 325,52 (-99,86%), uma estimativa
baseada em premissas e não uma medição em produção
Treino: cabe em ~3,36 GB de VRAM, com apenas 1,18% dos pesos treináveis
Próximo passo: acoplar o adaptador a Constrained Decoding (Guided JSON / Outlines) para
atacar os 25% residuais de falhas sintáticas, rumo a 99,9%+ de Schema Compliance.

Ficha Técnica
Stack de Engenharia Python, PyTorch, Hugging Face (transformers, peft, datasets,
bitsandbytes), Pydantic v2, Pytest, Docker, NVIDIA Tesla T4
Repositório GitHub LucasFig0/delivery-catalog-slm-qlora
Dataset (1.150 SKUs Reais LucasFig0/brazilian-delivery-catalog-sft
EAN-13)
Pesos do Adaptador LoRA LucasFig0/qwen2.5-1.5b-delivery-catalog-qlora
e Model Card
Sumário
1. O Problema de Engenharia
2. Engenharia de Dados
3. Contrato Pydantic v2 e Guardrail de Segurança Alimentar
4. Arquitetura de Treinamento
5. Resultados Experimentais
6. Unit Economics em Escala
7. Limitações
8. Conclusão e Próximos Passos
1. O Problema de Engenharia
Entropia Léxica e Unit Economics em Escala de Marketplace
1.1. Contexto: entropia léxica no catálogo
Em plataformas de delivery e supermercados online com dezenas de milhões de itens ativos, o
catálogo é alimentado continuamente por integrações heterogêneas de ERPs de varejo, PDVs de
restaurantes e cadastros manuais. O resultado operacional é uma alta entropia léxica:
Títulos truncados de nota fiscal: ACHOC PO TODDY 370G , AZEITE EXT VIRGEM ANDORINHA ,
LEITE PO NINHO INTEGRAL 380 G
Unidades de medida sem padronização: 1,5 l , 500 G , 350ml
Listas brutas de ingredientes em texto corrido, segundo o padrão regulatório da ANVISA
1.2. Consequências da falta de metadados estruturados
Sem metadados estruturados, o motor de busca sofre perda severa de Recall semântico e a
plataforma fica exposta a riscos críticos de segurança alimentar. Os metadados necessários são:
Categoria canônica
Proteína ou base principal
Tamanho da porção
Lista limpa de componentes

Sinônimos para expansão de busca
Tags de restrição alimentar: Sem Glúten, Sem Lactose, Zero Açúcar, Vegano, Vegetariano
1.3. Gargalos de produção
Escalar o enriquecimento de 50 milhões de SKUs mensais utilizando LLMs proprietários de
fronteira via API comercial esbarra em dois gargalos:
1. Inviabilidade de Unit Economics (COGS): Para forçar um modelo generalista a seguir
taxonomias estritas e regras de negócio sem alucinar chaves, são necessários prompts Few-
Shot extensos (~950 tokens de entrada e ~220 tokens de saída por SKU). A preços
comerciais de mercado (~US$ 2,50 por milhão de tokens de entrada e US$ 10,00 por milhão
de tokens de saída), o processamento mensal custa US$ 228.750,00/mês (US$ 2,74
milhões/ano).
2. Fragilidade de contrato e segurança alimentar em modelos menores Zero-Shot: Small
Language Models (SLMs de 1B a 3B de parâmetros) rodam com frações desse custo em
GPUs dedicadas, mas, quando avaliados em Zero-Shot, quebram frequentemente o esquema
JSON (envolvendo a saída em texto conversacional ou blocos Markdown) e cometem erros
graves de classificação biológica, como atribuir a tag Vegano a produtos de origem animal.
1.4. Solução proposta
Para resolver essa equação entre custo operacional, determinismo estrutural e segurança
alimentar, desenvolvi o projeto delivery-catalog-slm-qlora, aplicando Supervised Fine-Tuning
(SFT) via QLoRA (4-bit NormalFloat) no modelo Qwen/Qwen2.5-1.5B-Instruct sobre uma base
de 1.150 SKUs reais do mercado brasileiro.
2. Engenharia de Dados
Curadoria de 1.150 SKUs Reais Auditáveis por EAN-13
2.1. Por que evitar dados sintéticos
Uma armadilha comum em benchmarks de Fine-Tuning é treinar e avaliar o modelo sobre dados
sintéticos gerados por regras ou templates. Um SLM de 1,5 bilhão de parâmetros memoriza
padrões sintéticos em poucas dezenas de steps, inflando o F1-Score para perto de 100% em
validação, mas colapsando ao encontrar a sujeira real de um ERP de supermercado ou restaurante.
Para garantir validade ecológica e reprodutibilidade científica, construímos um pipeline
Medallion (Bronze → Silver → Gold) fundamentado 100% em produtos reais comercializados no
Brasil.
2.2. Camada Bronze ( src/raw_ingestion.py )
Ingestão via API do registro público Open Food Facts Brasil, combinando itens de alto giro
do varejo nacional com amostragem estratificada em 25 partições gastronômicas (pizzas,
sandwiches, burgers, meals, meats, poultry, seafood, fishes, snacks, beverages, dairies,
desserts, breads, plant-based-foods, entre outras).
Cada registro exige código de barras EAN-13 real (789...), título original de
embalagem/PDV e declaração real de ingredientes em português.
Total: 1.250 SKUs brutos em data/raw/openfoodfacts_br_raw.jsonl .

| 2.3. Camada Silver/Gold ( | src/dataset_builder.py | )   |
| ------------------------- | ---------------------- | --- |
O transformador determinístico processa as listas reais de ingredientes da ANVISA com um
parser com controle de profundidade de escopo, que ignora vírgulas aninhadas dentro de
parênteses e colchetes químicos.
Exemplo: isola corretamente  MINERAIS [CÁLCIO (CARBONATO DE CÁLCIO), FERRO E
ZINCO]  como um componente coeso.
Descarta sufixos regulatórios de alerta ( ALÉRGICOS: CONTÉM... ).
2.4. Prevenção de Data Leakage
Os 1.150 SKUs validados são particionados sem qualquer sobreposição de código EAN-13 ou
texto de prompt, e publicados no Hugging Face Hub em formato ChatML (system, user,
assistant):
| Split      |     | SKUs |
| ---------- | --- | ---- |
| Train      |     | 800  |
| Validation |     | 150  |
| Test       |     | 200  |
3. Contrato Pydantic v2 e Guardrail de Segurança Alimentar
3.1. Princípio
Em sistemas de IA em produção, o modelo generativo nunca deve escrever diretamente no banco
de dados ou no índice de busca sem uma camada intermediária de validação formal. Todo o
Ground Truth de treino e todas as predições de inferência são submetidos ao schema
EnrichedCatalogItem , construído em Pydantic v2 ( src/schemas.py ):
python

class EnrichedCatalogItem(BaseModel):
model_config = ConfigDict(extra="forbid")
normalized_title: str = Field(..., min_length=3, max_length=120)
category: CatalogCategory
main_protein_or_base: MainProteinOrBase
portion_size: PortionSize
dietary_tags: list[DietaryTag] = Field(default_factory=list)
is_combo: bool
volume_or_weight: Optional[str] = Field(default=None)
components: list[str] = Field(..., min_length=1, max_length=12)
search_synonyms: list[str] = Field(..., min_length=2, max_length=6)
@model_validator(mode="after")
def validate_food_safety_consistency(self) -> "EnrichedCatalogItem":
if self.main_protein_or_base in ANIMAL_PROTEINS:
forbidden = {"Vegano",
"Vegetariano"}.intersection(set(self.dietary_tags))
if forbidden:
raise ValueError(
f"Violação de segurança alimentar: proteína
'{self.main_protein_or_base}' "
f"é incompatível com as tags {sorted(forbidden)}."
)
return self
3.2. Mecanismos de proteção
ConfigDict(extra="forbid") : rejeita imediatamente qualquer chave inventada pelo
modelo.
@model_validator : atua como trava determinística de segurança alimentar. Se a base
proteica identificada for animal (Bovino, Frango, Suíno ou Peixe e Frutos do Mar), a
presença de Vegano ou Vegetariano em dietary_tags dispara uma exceção de validação
antes que o registro contamine o catálogo.
4. Arquitetura de Treinamento
QLoRA 4-bit NF4 e Completion-Only Loss Masking
O treinamento foi projetado para executar integralmente dentro do envelope de memória de uma
única GPU NVIDIA Tesla T4 (16 GB VRAM). Para maximizar a capacidade de adaptação
estrutural sem sofrer Out-Of-Memory (OOM), combinamos três técnicas no módulo
src/qlora_config.py .

4.1. Quantização NormalFloat 4-bit (NF4) com Double Quantization
d×k do Qwen/Qwen2.5-1.5B-Instruct são congelados e
| Os pesos pré-treinados W |     |     | ∈   |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|                          |     |     | 0   | R   |     |     |     |     |     |     |

quantizados no tipo de dado informacionalmente ótimo para distribuições normais centradas em
zero ( ), com quantização dupla das constantes de escala. Isso comprime os 1,54 bilhão de
nf4
parâmetros base para apenas 0,81 GB de VRAM.
4.2. Cobertura Total das Projeções Lineares (Atenção + MLP)
Em vez de aplicar adaptadores apenas nas matrizes de Query e Value (W ,W ), estendemos os
q v

adaptadores LoRA (r = 16,α = 32,dropout = 0.05) para todas as 7 projeções lineares de cada
um dos 28 blocos Transformer ( q_proj ,  k_proj ,  v_proj ,  o_proj ,  gate_proj ,  up_proj ,
| down_proj | ).  |     |     |     |     |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Durante o forward pass, a atualização de uma camada linear para uma entrada x é dada por:
α
|        |         |     | h                                       | = W | x + ΔWx |     | = W | x + | BAx |     |
| ------ | ------- | --- | --------------------------------------- | --- | ------- | --- | --- | --- | --- | --- |
|        |         |     |                                         |     | 0       |     |     | 0   |     |     |
|        |         |     |                                         |     |         |     |     |     | r   |     |
|        | r×k e B |     | d×r são matrizes treináveis em float16. |     |         |     |     |     |     |     |
| onde A | ∈ R     | ∈   | R                                       |     |         |     |     |     |     |     |
Parâmetros e memória:
| Item                  |     |     | Valor |     |         |           |     |     |     |     |
| --------------------- | --- | --- | ----- | --- | ------- | --------- | --- | --- | --- | --- |
| Dimensões do Qwen2.5- |     |     | d     | =   | 1536, d | = 8960, d |     | =   | 256 |     |
|                       |     |     | model |     |         | ffn       |     | kv  |     |     |
|                       |     |     |       |     |         |           |     |     |     |     |
1.5B
| Parâmetros treináveis |     |     | 18.464.768    |     |     |     |     |     |     |     |
| --------------------- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
| Total de parâmetros   |     |     | 1.562.179.072 |     |     |     |     |     |     |     |
| Fração treinável      |     |     | 1,1820%       |     |     |     |     |     |     |     |
Consumo total estimado ~3,36 GB (pesos NF4 + otimizador paginado  paged_adamw_8bit  +
| de VRAM |     |     | ativações com Gradient Checkpointing) |     |     |     |     |     |     |     |
| ------- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
4.3. Completion-Only Loss Masking
Em tarefas de extração estruturada, calcular a perda autoregressiva sobre os tokens do prompt
(system + user) desperdiça capacidade de representação dos adaptadores LoRA tentando prever
o texto ruidoso do restaurante. Na função de tokenização do pipeline, mascaramos todos os
tokens de contexto com   (o   padrão da função de perda do PyTorch),
|     |     |     | -100 | ignore_index |     |     |     |     |     |     |
| --- | --- | --- | ---- | ------------ | --- | --- | --- | --- | --- | --- |
restringindo o gradiente estritamente aos tokens da resposta JSON do assistant (y  para t ≥
t

| ∣x  | ∣): |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
prompt

T
|     |     | L   | (Θ  |      | ) = − |      | logP |     | (y ∣ y  | )   |
| --- | --- | --- | --- | ---- | ----- | ---- | ---- | --- | ------- | --- |
|     |     |     | SFT | LoRA |       |      |      | Θ   | t 1…t−1 |     |
|     |     |     |     |      |       | ∑    |      |     |         |     |
|     |     |     |     |      |       | t=∣x | ∣    |     |         |     |
prompt

5. Resultados Experimentais
Holdout de 200 SKUs Reais Inéditos, GPU T4
5.1. Protocolo de avaliação
Para garantir uma comparação Apples-to-Apples, avaliamos os 200 SKUs reais do conjunto de
Holdout (test) em duas condições, sob o mesmo SYSTEM_PROMPT e decodificação determinística
( do_sample=False ):
Condição Descrição
Baseline Zero- Qwen/Qwen2.5-1.5B-Instruct quantizado em 4-bit NF4, sem ajuste fino
Shot
Fine-Tuned Mesmo modelo base acoplado ao adaptador LoRA treinado por 2 épocas (100 steps
QLoRA de atualização, batch efetivo de 16, lr=2e-4, scheduler cosseno)

5.2. Tabela comparativa
Métrica de Engenharia Base Fine-Tuned Delta Impacto em Produção
| (Holdout n=200) | Zero-Shot | QLoRA |     |     |
| --------------- | --------- | ----- | --- | --- |
JSON Schema 18,50% 75,00% +56,50 4,05x mais conformidade Pydantic
| Compliance Rate (%) |     |     | p.p. | v2  |
| ------------------- | --- | --- | ---- | --- |
Strict Clean JSON 0,00% 60,50% +60,50 Emissão direta sem blocos
| (Sem Reparo) (%) |     |     | p.p. | Markdown |
| ---------------- | --- | --- | ---- | -------- |
Mean Field Exact 9,60% 69,70% +60,10 7,26x de ganho nos campos
| Match (%) |     |     | p.p. | escalares |
| --------- | --- | --- | ---- | --------- |
Category Accuracy (%) 5,50% 75,00% +69,50 100,0% condicional nos JSONs
|     |     |     | p.p. | válidos |
| --- | --- | --- | ---- | ------- |
Main Protein/Base 3,00% 61,50% +58,50 82,0% condicional nos JSONs
| Accuracy (%) |     |     | p.p. | válidos |
| ------------ | --- | --- | ---- | ------- |
Portion Size Accuracy 5,50% 64,50% +59,00 86,0% condicional nos JSONs
| (%) |     |     | p.p. | válidos |
| --- | --- | --- | ---- | ------- |
Dietary Tags Micro-F1 10,22% 31,84% +21,62 3,11x de ganho multi-label
| (%) |     |     | p.p. | nutricional, ainda baixo em termos |
| --- | --- | --- | ---- | ---------------------------------- |
absolutos
Components Token F1 12,57% 67,79% +55,22 90,39% de F1 condicional
| (%) |     |     | p.p. | (ANVISA) |
| --- | --- | --- | ---- | -------- |
Search Synonyms 4,38% 63,97% +59,59 85,29% de F1 condicional para
| Token F1 (%) |     |     | p.p. | busca |
| ------------ | --- | --- | ---- | ----- |
Food Safety Violations 11 (5,5%) 0 (0,0%) -11 Nenhuma violação no holdout (as
| (Count) |     |     |     | 11 do baseline eliminadas) |
| ------- | --- | --- | --- | -------------------------- |
Latency P50 (ms/SKU) 1123,61 1934,95 ms* +811,34 *Com 196 hooks PEFT não-
|     | ms  |     | ms  | fundidos |
| --- | --- | --- | --- | -------- |
5.3. Acurácia Global Penalizada vs. Acurácia Condicional
O motor de avaliação ( src/evaluator.py ) aplica uma regra severa de engenharia de produção:
se a resposta do modelo falhar no parser JSON ou violar qualquer restrição do
 (Pydantic v2), ela recebe pontuação zero em todos os atributos daquela
EnrichedCatalogItem
amostra.
Por isso, além do salto global de 4,05x em conformidade de contrato (18,50% para 75,00%), é
fundamental analisar a acurácia condicional dentro dos 150 SKUs (75,00%) em que o contrato

JSON foi validado com sucesso. A acurácia condicional mostra a qualidade do modelo quando o
contrato é cumprido, mas não substitui a métrica global, que é a que reflete o comportamento em
produção:
Métrica Cálculo Resultado condicional
Category Accuracy 75,00% 100,0% (150 de 150 JSONs válidos)
75,00%
Components Token 67,79% 90,39% de precisão/revocação lexical na higienização da lista real
75,00%
F1 de ingredientes da ANVISA
Portion Size Accuracy 64,50% 86,00%
75,00%
Search Synonyms 63,97% 85,29%
75,00%
Token F1
5.4. Violações de Segurança Alimentar
No baseline Zero-Shot, o modelo cometeu 11 violações críticas de segurança alimentar (5,5%
das amostras de teste), atribuindo tags Vegano ou Vegetariano a itens que continham proteína
animal real. Após o SFT com QLoRA alinhado às regras do dataset real, o número de violações
caiu para zero (0,00%) nos 200 SKUs de Holdout. Essa métrica deve ser lida junto com a taxa de
conformidade de 75%: respostas fora do contrato não chegam ao catálogo.
5.5. Análise de Sistemas: por que o PEFT não-fundido aumenta a latência no
benchmark?
A latência mediana por SKU subiu de 1123,61 ms para 1934,95 ms na avaliação direta do objeto
PeftModel em PyTorch. Esse comportamento decorre de dois fatores mecânicos:
1. Overhead de 196 hooks dinâmicos PEFT sobre pesos NF4: Com adaptadores anexados a 7
módulos lineares em 28 camadas Transformer, cada passo de geração autoregressiva
executa 196 chamadas extras de kernels CUDA para calcular αBAx em paralelo à
r
desquantização on-the-fly dos blocos NF4.
2. Solução de produção ( merge_and_unload + vLLM/TGI): Em deploy produtivo, nunca
servimos os adaptadores LoRA desacoplados. Executamos a fusão algébrica offline
(W = W + αBA) via model.merge_and_unload() , exportando um checkpoint
merged 0 r
único para motores com PagedAttention e Continuous Batching (como vLLM ou TensorRT-
LLM). Isso zera completamente o overhead de latência dos adaptadores e reduz a latência
por SKU em lote para menos de 35 ms. Esse valor é uma projeção e não foi medido no
benchmark acima.
6. Unit Economics em Escala
50 Milhões de SKUs/Mês (modelo de custos)
6.1. Premissas do modelo
Modelando uma operação de catálogo de larga escala que processa 50 milhões de atualizações
de SKUs por mês ( simulate_unit_economics em src/evaluator.py ), comparamos o custo de

uma API comercial de LLM de fronteira contra o Qwen2.5-1.5B-Instruct pós-QLoRA, fundido e
servido em instância GPU dedicada:
| Premissa | API LLM Proprietária                     |     | SLM QLoRA Self-Hosted |     |
| -------- | ---------------------------------------- | --- | --------------------- | --- |
| Prompt   | Few-Shot: 950 tokens de entrada e 220 de |     | n/d                   |     |
saída
Preço ~US$ 2,50/M tokens de entrada e US$ 10,00/M US$ 0,75/hora (instância de nuvem)
tokens de saída
| Throughput | n/d |     | 32 SKUs/s (conservador, sob |     |
| ---------- | --- | --- | --------------------------- | --- |
continuous batching)
6.2. Resultados do modelo
Indicador Operacional e API LLM Proprietária SLM QLoRA Self- Economia /
| Financeiro |     | (Few-Shot) | Hosted (GPU) | Impacto |
| ---------- | --- | ---------- | ------------ | ------- |
Volume Mensal de 50.000.000 SKUs 50.000.000 SKUs Escala Nacional
Catálogo
| Custo por 1.000 SKUs |     | US$ 4,5750 | US$ 0,0065 | -99,86% por lote |
| -------------------- | --- | ---------- | ---------- | ---------------- |
Horas de GPU N/A (Serverless API) 434,03 horas (~0,6 1 instância T4/L4
| Necessárias/Mês |     |     | GPU) | dedicada |
| --------------- | --- | --- | ---- | -------- |
Custo Mensal Total US$ 228.750,00 US$ 325,52 US$ 228.424,48 /
| (COGS)              |     |     |     | mês                |
| ------------------- | --- | --- | --- | ------------------ |
| Economia Anualizada |     | n/d | n/d | US$ 2.741.093,75 / |
| Projetada           |     |     |     | ano                |
6.3. Como interpretar estes números
Os valores acima são uma estimativa baseada em premissas, e não uma medição em produção.
Eles consideram apenas o custo de inferência (tokens da API versus horas de GPU), com
throughput e preços assumidos. Não incluem custo de engenharia e manutenção, nem o
tratamento dos 25% de respostas que não passam no contrato. Além disso, a qualidade do LLM de
fronteira não foi avaliada neste trabalho, então a comparação é de custo e não de qualidade
equivalente.
7. Limitações
Tags dietéticas ainda fracas: o Micro-F1 de Dietary Tags é de 31,84%. O guardrail
determinístico cobre apenas a incompatibilidade entre proteína animal e as tags

Vegano/Vegetariano; as demais tags (Sem Glúten, Sem Lactose, Zero Açúcar) dependem da
acurácia do modelo.
25% de falhas de contrato: um quarto das respostas ainda quebra o JSON ou viola o
schema, majoritariamente por listas de ingredientes extensas. Em produção isso exigiria
retry ou fallback, o que o modelo de custos da seção 6 não considera.
Amostra e protocolo reduzidos: 1.150 SKUs no total, 200 no teste e 100 steps de treino. Os
resultados correspondem a uma única configuração reportada, sem intervalos de confiança
ou repetição com múltiplas seeds.
Latência medida versus projetada: a latência medida com o adaptador não fundido foi de
1934,95 ms por SKU. O valor abaixo de 35 ms após merge_and_unload com vLLM é uma
projeção, não um resultado do benchmark.
Economia é uma estimativa: o custo mensal e a economia anualizada derivam de
premissas de preço e throughput, sem medição em produção e sem comparação de
qualidade com o LLM de fronteira.
Origem dos dados: os SKUs vêm do Open Food Facts Brasil, um registro público, e não de
feeds reais de ERP ou PDV de um marketplace, então a sujeira de um ambiente real pode
diferir.
8. Conclusão e Próximos Passos Arquiteturais
8.1. Conclusão
Os resultados empíricos em GPU T4 indicam que um Small Language Model de 1,5 bilhão de
parâmetros, quando ajustado via QLoRA (1,18% de pesos treináveis) sobre dados reais curados
por EAN-13 e governado por contratos Pydantic v2:
Supera em 4,05x a conformidade estrutural do modelo base Zero-Shot (de 18,50% para
75,00%)
Atinge 100% de precisão condicional na taxonomia de categorias, dentro dos JSONs
válidos
Zera as violações de segurança alimentar observadas no holdout
Reduz o COGS modelado de processamento de catálogo em 99,86%
8.2. Próximo passo: Constrained Decoding
Para elevar os 25% residuais de falhas sintáticas (causadas majoritariamente por listas de
ingredientes muito extensas que truncam ou quebram aspas em geração livre) rumo a 99,9%+ de
Schema Compliance em produção, o próximo passo natural no servidor de inferência (vLLM /
SGLang) é acoplar os pesos do adaptador LoRA a Constrained Decoding por Máscara de Logits
(Guided JSON / Outlines FSM), compilando o JSON Schema do EnrichedCatalogItem em um
autômato finito que impede matematicamente a amostragem de tokens fora da gramática JSON.