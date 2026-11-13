from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.schemas import (
    ANIMAL_PROTEINS,
    CatalogCategory,
    DietaryTag,
    EnrichedCatalogItem,
    MainProteinOrBase,
    PortionSize,
    RawCatalogItem,
)

SYSTEM_PROMPT = (
    "Você é um motor especialista em Inteligência de Catálogo e Busca do iFood. "
    "Sua tarefa é receber um SKU real de cardápio ou mercado (código EAN, loja, título bruto e "
    "lista de ingredientes/descrição) e retornar ESTRITAMENTE um objeto JSON válido "
    "(sem texto extra e sem blocos markdown) seguindo o esquema exato:\n"
    "{\n"
    '  "normalized_title": str,\n'
    '  "category": "Lanches" | "Pizzas" | "Brasileira" | "Japonesa" | "Açaí e Sorvetes" | '
    '"Bebidas" | "Sobremesas" | "Saudável e Fit" | "Padaria e Cafeteria" | "Mercado",\n'
    '  "main_protein_or_base": "Bovino" | "Frango" | "Suíno" | "Peixe e Frutos do Mar" | '
    '"Vegetariano/Vegano" | "Massa/Queijo" | "Açaí/Frutas" | "Bebida/Outro",\n'
    '  "portion_size": "Individual" | "Casal (2 Pessoas)" | "Família (3+ Pessoas)" | '
    '"Porção/Acompanhamento" | "Unitário",\n'
    '  "dietary_tags": list["Zero Açúcar" | "Sem Lactose" | "Sem Glúten" | '
    '"Vegano" | "Vegetariano" | "Low Carb/Fit"],\n'
    '  "is_combo": bool,\n'
    '  "volume_or_weight": str | null,\n'
    '  "components": list[str],\n'
    '  "search_synonyms": list[str]\n'
    "}\n"
    "Regra crítica de segurança alimentar: nunca atribua 'Vegano' ou 'Vegetariano' "
    "a itens com proteína animal (Bovino, Frango, Suíno, Peixe e Frutos do Mar)."
)


def extract_clean_measure(quantity_raw: str, title_raw: str) -> Optional[str]:
    """Extrai e padroniza medidas reais (g, kg, ml, L, un) da embalagem ou título."""
    for source in (quantity_raw, title_raw):
        if not source:
            continue
        cleaned = source.replace(",", ".")
        match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(ml|l|litros?|g|gramas?|kg|un|und|unidades?)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if match:
            val_str, unit_raw = match.group(1), match.group(2).lower()
            if unit_raw.startswith("l"):
                unit = "L"
            elif unit_raw.startswith("g"):
                unit = "g"
            elif unit_raw.startswith("un"):
                unit = "un"
            else:
                unit = unit_raw
            try:
                val_num = float(val_str)
                if val_num <= 0 or val_num > 25000:
                    continue
                if val_num.is_integer():
                    return f"{int(val_num)}{unit}"
                return f"{val_num}{unit}"
            except ValueError:
                continue
    return None


def extract_real_components(ingredients_raw: str, fallback_title: str) -> list[str]:
    """
    Faz o parsing da lista real de ingredientes (padrão ANVISA),
    ignorando vírgulas dentro de parênteses/colchetes.
    """
    text = re.sub(r"^ingredientes\s*[:\-]?\s*", "", ingredients_raw.strip(), flags=re.IGNORECASE)
    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for ch in text:
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch in ",;." and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
        else:
            current.append(ch)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    cleaned_components: list[str] = []
    stop_prefixes = ("alérgicos", "alergicos", "contém", "contem", "pode conter", "não contém", "nao contem")

    for raw_part in parts:
        part = re.sub(r"\([^)]*\)", "", raw_part)
        part = re.sub(r"\[[^\]]*\]", "", part)
        part = re.sub(r"\s+", " ", part).strip(" -:*_/")
        if len(part) < 3 or len(part) > 65:
            continue
        if part.lower().startswith(stop_prefixes):
            continue
        formatted = part[0].upper() + part[1:].lower() if part.isupper() else part
        if formatted not in cleaned_components:
            cleaned_components.append(formatted)
        if len(cleaned_components) >= 6:
            break

    if not cleaned_components:
        clean_fb = re.sub(r"\s+", " ", fallback_title).strip()[:60]
        cleaned_components.append(clean_fb if len(clean_fb) >= 3 else "Ingrediente principal")

    return cleaned_components


def infer_category_and_merchant(
    title: str,
    categories_text: str,
    categories_tags: list[str],
    ingredients: str,
) -> tuple[CatalogCategory, str]:
    """Classifica a categoria canônica e o tipo de estabelecimento a partir dos metadados reais."""
    blob = f"{title} {categories_text} {' '.join(categories_tags)} {ingredients}".lower()

    if any(k in blob for k in ("pizza", "calzone", "pizzaria")):
        return "Pizzas", "Pizzaria e Massas"
    if any(k in blob for k in ("sushi", "sashimi", "temaki", "yakisoba", "shoyu", "nori", "wasabi", "oriental")):
        return "Japonesa", "Restaurante Oriental e Conveniência"
    if any(k in blob for k in ("açaí", "acai", "sorvete", "gelato", "picolé", "picole", "frozen-desserts")):
        return "Açaí e Sorvetes", "Açaiteria e Sorveteria"
    if any(
        k in blob
        for k in (
            "hambúrguer",
            "hamburguer",
            "burger",
            "sanduíche",
            "sanduiche",
            "coxinha",
            "kibe",
            "empada",
            "pão de queijo",
            "pao de queijo",
            "salgadinho",
            "batata frita",
            "nuggets",
            "empanado",
            "snacks",
            "sandwiches",
            "burgers",
        )
    ):
        return "Lanches", "Lanchonete e Conveniência"
    if any(
        k in blob
        for k in (
            "refrigerante",
            "suco",
            "néctar",
            "nectar",
            "bebida",
            "água mineral",
            "agua mineral",
            "cerveja",
            "energético",
            "energetico",
            "chá ",
            "cha ",
            "beverages",
            "carbonated-drinks",
            "juices",
        )
    ):
        return "Bebidas", "Bebidas e Conveniência"
    if any(
        k in blob
        for k in (
            "chocolate",
            "bombom",
            "brigadeiro",
            "sobremesa",
            "doce de leite",
            "goiabada",
            "pudim",
            "gelatina",
            "bala ",
            "trufa",
            "desserts",
            "chocolates",
        )
    ):
        return "Sobremesas", "Doceria e Confeitaria"
    if any(
        k in blob
        for k in (
            "pão ",
            "pao ",
            "bolo ",
            "biscoito",
            "bolacha",
            "torrada",
            "panetone",
            "croissant",
            "tapioca",
            "café",
            "cafe",
            "achocolatado",
            "cappuccino",
            "breads",
            "biscuits-and-cakes",
        )
    ):
        return "Padaria e Cafeteria", "Padaria e Cafeteria"
    if any(
        k in blob
        for k in (
            "integral",
            "granola",
            "aveia",
            "whey",
            "proteico",
            "proteína",
            "chia",
            "linhaça",
            "linhaca",
            "castanha",
            "amendoim",
            "plant-based",
            "vegano",
        )
    ):
        return "Saudável e Fit", "Empório Saudável e Fit"
    if any(
        k in blob
        for k in (
            "feijoada",
            "lasanha",
            "escondidinho",
            "parmegiana",
            "strogonoff",
            "estrogonofe",
            "prato pronto",
            "refeição",
            "picanha",
            "carne de sol",
            "linguiça",
            "linguica",
            "frango",
            "bovino",
            "meals",
            "meats",
            "poultry",
        )
    ):
        return "Brasileira", "Restaurante e Rotisseria"

    return "Mercado", "Mercado e Mercearia Express"


def infer_main_protein_or_base(
    title: str,
    ingredients: str,
    categories_tags: list[str],
    category: CatalogCategory,
) -> MainProteinOrBase:
    """Identifica a proteína ou base principal real do SKU."""
    title_cats = f"{title} {' '.join(categories_tags)}".lower()
    ing_lower = ingredients.lower()

    if any(k in title_cats for k in ("salmão", "salmao", "atum", "bacalhau", "tilápia", "tilapia", "camarão", "camarao", "sardinha", "peixe", "seafood", "fishes")):
        return "Peixe e Frutos do Mar"
    if any(k in title_cats for k in ("frango", "peru", "chester", "galinha", "chicken", "poultry")):
        return "Frango"
    if any(k in title_cats for k in ("calabresa", "bacon", "presunto", "lombo", "pernil", "suíno", "suino", "linguiça", "linguica", "mortadela", "salsicha", "peito de peru")):
        return "Suíno"
    if any(k in title_cats for k in ("bovino", "bovina", "picanha", "alcatra", "costela", "carne moída", "carne de sol", "charque", "carne bovina", "hambúrguer bovino")):
        return "Bovino"

    if any(k in ing_lower for k in ("carne mecanicamente separada de ave", "peito de frango", "carne de frango")):
        return "Frango"
    if any(k in ing_lower for k in ("carne suína", "carne suina", "toucinho", "barriga suína", "pernil suíno")):
        return "Suíno"
    if any(k in ing_lower for k in ("carne bovina",)):
        return "Bovino"
    if any(k in ing_lower for k in ("salmão", "atum", "sardinha", "camarão", "merluza", "tilápia")):
        return "Peixe e Frutos do Mar"

    if category == "Açaí e Sorvetes" or any(k in title_cats for k in ("açaí", "acai", "fruta", "polpa", "morango", "banana")):
        return "Açaí/Frutas"
    if category == "Bebidas":
        return "Bebida/Outro"
    if any(k in title_cats for k in ("vegano", "vegan", "plant-based", "soja", "grão-de-bico", "cogumelo", "vegetal")):
        return "Vegetariano/Vegano"
    if any(k in title_cats for k in ("queijo", "mussarela", "muçarela", "requeijão", "massa", "macarrão", "pão", "pao", "biscoito", "bolo", "pizza", "lasanha")):
        return "Massa/Queijo"

    return "Bebida/Outro"


def infer_portion_size(measure: Optional[str], title: str) -> PortionSize:
    """Estima o tamanho da porção a partir do peso/volume real do produto."""
    title_l = title.lower()
    if any(k in title_l for k in ("família", "familia", "gigante", "econômica", "economica")):
        return "Família (3+ Pessoas)"
    if any(k in title_l for k in ("casal", "2 pessoas", "duplo")):
        return "Casal (2 Pessoas)"
    if any(k in title_l for k in ("porção", "porcao", "aperitivo", "petisco")):
        return "Porção/Acompanhamento"

    if measure:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)(ml|L|g|kg|un)", measure)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            if unit == "kg" or (unit == "g" and val >= 900) or (unit == "L" and val >= 1.2) or (unit == "ml" and val >= 1200):
                return "Família (3+ Pessoas)"
            if (unit == "g" and 450 <= val < 900) or (unit == "ml" and 600 <= val < 1200):
                return "Casal (2 Pessoas)"
            if (unit == "g" and 120 <= val < 450) or (unit == "ml" and 200 <= val < 600):
                return "Individual"

    return "Unitário"


def infer_dietary_tags(
    title: str,
    ingredients: str,
    labels_tags: list[str],
    analysis_tags: list[str],
    allergens_tags: list[str],
    main_protein: MainProteinOrBase,
) -> list[DietaryTag]:
    """Extrai tags dietéticas reais respeitando a trava de segurança alimentar."""
    tags: list[DietaryTag] = []
    text_blob = f"{title} {ingredients}".lower()
    labels_set = {t.lower() for t in (labels_tags or [])}
    analysis_set = {t.lower() for t in (analysis_tags or [])}
    allergens_set = {t.lower() for t in (allergens_tags or [])}

    if (
        "en:no-added-sugar" in labels_set
        or "en:sugar-free" in labels_set
        or any(k in text_blob for k in ("zero açúcar", "zero acucar", "sem açúcar", "sem acucar", "sem adição de açúcares", "zero adição de açúcares", "diet"))
    ):
        tags.append("Zero Açúcar")

    has_milk_allergen = "en:milk" in allergens_set
    if (
        "en:no-lactose" in labels_set
        or "en:lactose-free" in labels_set
        or any(k in text_blob for k in ("zero lactose", "sem lactose", "0% lactose", "não contém lactose", "nao contem lactose"))
    ):
        tags.append("Sem Lactose")

    has_gluten_allergen = "en:gluten" in allergens_set
    explicit_contains_gluten = (
        any(k in text_blob for k in ("contém glúten", "contem gluten"))
        and not any(k in text_blob for k in ("não contém glúten", "nao contem gluten"))
    )
    if (
        ("en:no-gluten" in labels_set or "en:gluten-free" in labels_set or any(k in text_blob for k in ("não contém glúten", "nao contem gluten", "sem glúten", "sem gluten")))
        and not explicit_contains_gluten
        and not has_gluten_allergen
    ):
        tags.append("Sem Glúten")

    if main_protein not in ANIMAL_PROTEINS:
        is_vegan = (
            "en:vegan" in labels_set
            or "en:vegan" in analysis_set
            or any(k in text_blob for k in ("100% vegano", "produto vegano", "vegano"))
        )
        if is_vegan and not has_milk_allergen:
            tags.append("Vegano")
            tags.append("Vegetariano")
        elif "en:vegetarian" in labels_set or "en:vegetarian" in analysis_set:
            tags.append("Vegetariano")

    if any(k in text_blob for k in ("low carb", "proteico", "whey", "integral", "zero açúcar", "fit")):
        tags.append("Low Carb/Fit")

    dedup: list[DietaryTag] = []
    for t in tags:
        if t not in dedup:
            dedup.append(t)
    return dedup


def normalize_real_title(raw_title: str, brand: str, measure: Optional[str]) -> str:
    """Higieniza o título bruto do PDV/ERP mantendo a fidelidade ao produto real."""
    clean = re.sub(r"\s+", " ", raw_title).strip(" -_,./|")
    if clean.isupper() or clean.islower():
        clean = clean.title()

    lower_words = {"De", "Da", "Do", "Das", "Dos", "Com", "Sem", "Em", "Para", "E", "Ou"}
    words = clean.split(" ")
    fixed_words = [
        w.lower() if (idx > 0 and w in lower_words) else w
        for idx, w in enumerate(words)
    ]
    clean = " ".join(fixed_words)

    brand_first = brand.split(",")[0].strip() if brand else ""
    if brand_first and len(brand_first) >= 2 and brand_first.lower() not in clean.lower():
        if len(clean) + len(brand_first) + 1 <= 100:
            clean = f"{clean} {brand_first.title()}"

    if measure and measure.lower() not in clean.lower():
        if len(clean) + len(measure) + 1 <= 115:
            clean = f"{clean} {measure}"

    return clean[:118].strip()


def build_search_synonyms(
    normalized_title: str,
    brand: str,
    category: CatalogCategory,
    generic_name: str,
    dietary_tags: list[DietaryTag],
) -> list[str]:
    """Gera termos reais de busca baseados na marca, categoria, nome genérico e restrições."""
    synonyms: list[str] = []

    if generic_name and len(generic_name.strip()) >= 3:
        synonyms.append(generic_name.strip().lower()[:50])

    brand_clean = brand.split(",")[0].strip().lower() if brand else ""
    first_word = normalized_title.split(" ")[0].lower()
    if brand_clean and len(brand_clean) >= 2:
        synonyms.append(f"{first_word} {brand_clean}".strip()[:50])

    synonyms.append(f"{category.lower()} delivery")

    if dietary_tags:
        synonyms.append(f"{first_word} {dietary_tags[0].lower()}")
    else:
        synonyms.append(f"comprar {first_word}")

    cleaned: list[str] = []
    for s in synonyms:
        s_norm = re.sub(r"\s+", " ", s).strip()
        if len(s_norm) >= 3 and s_norm not in cleaned:
            cleaned.append(s_norm)

    while len(cleaned) < 2:
        fallback = f"item de {category.lower()}" if len(cleaned) == 0 else "catalogo ifood"
        if fallback not in cleaned:
            cleaned.append(fallback)

    return cleaned[:5]


def format_user_prompt(raw_item: RawCatalogItem) -> str:
    """Formata o payload real do SKU para inferência no SLM."""
    return (
        f"SKU_ID: {raw_item.sku_id}\n"
        f"LOJA: {raw_item.merchant_type}\n"
        f"TITULO_BRUTO: {raw_item.raw_title}\n"
        f"DESCRICAO_INGREDIENTES: {raw_item.raw_description}\n"
        f"PRECO_ESTIMADO_BRL: R$ {raw_item.price_brl:.2f}"
    )


def transform_raw_off_record(raw_prod: dict[str, Any], rng: random.Random) -> Optional[dict[str, Any]]:
    """Converte um registro bruto real do Open Food Facts BR em exemplo SFT validado."""
    code = str(raw_prod.get("code") or "").strip()
    raw_name = (raw_prod.get("product_name_pt") or raw_prod.get("product_name") or "").strip()
    ingredients = (raw_prod.get("ingredients_text_pt") or raw_prod.get("ingredients_text") or "").strip()
    brand = str(raw_prod.get("brands") or "").strip()
    quantity = str(raw_prod.get("quantity") or "").strip()
    generic_name = str(raw_prod.get("generic_name_pt") or "").strip()
    categories_text = str(raw_prod.get("categories") or "").strip()
    categories_tags = list(raw_prod.get("categories_tags") or [])
    labels_tags = list(raw_prod.get("labels_tags") or [])
    analysis_tags = list(raw_prod.get("ingredients_analysis_tags") or [])
    allergens_tags = list(raw_prod.get("allergens_tags") or [])

    if not code or len(raw_name) < 3 or len(ingredients) < 10:
        return None

    measure = extract_clean_measure(quantity, raw_name)
    category, merchant_type = infer_category_and_merchant(
        raw_name, categories_text, categories_tags, ingredients
    )
    main_protein = infer_main_protein_or_base(raw_name, ingredients, categories_tags, category)
    portion_size = infer_portion_size(measure, raw_name)
    dietary_tags = infer_dietary_tags(
        raw_name, ingredients, labels_tags, analysis_tags, allergens_tags, main_protein
    )
    components = extract_real_components(ingredients, raw_name)
    normalized_title = normalize_real_title(raw_name, brand, measure)
    is_combo = any(k in raw_name.lower() for k in ("combo", "kit ", "combinado", " + ", " c/ refri"))
    search_synonyms = build_search_synonyms(
        normalized_title, brand, category, generic_name, dietary_tags
    )

    raw_title_display = raw_name
    if quantity and quantity.lower() not in raw_name.lower():
        raw_title_display = f"{raw_name} - {quantity}"
    raw_title_display = raw_title_display[:195]

    desc_parts = []
    if brand:
        desc_parts.append(f"Marca: {brand}.")
    if generic_name:
        desc_parts.append(f"Descrição: {generic_name}.")
    desc_parts.append(f"Ingredientes: {ingredients}")
    raw_description_display = re.sub(r"\s+", " ", " ".join(desc_parts)).strip()[:590]

    estimated_price = round(rng.uniform(6.90, 89.90), 2)

    try:
        raw_item = RawCatalogItem(
            sku_id=f"EAN-{code}",
            raw_title=raw_title_display,
            raw_description=raw_description_display,
            merchant_type=merchant_type,
            price_brl=estimated_price,
        )
        gt_item = EnrichedCatalogItem(
            normalized_title=normalized_title,
            category=category,
            main_protein_or_base=main_protein,
            portion_size=portion_size,
            dietary_tags=dietary_tags,
            is_combo=is_combo,
            volume_or_weight=measure,
            components=components,
            search_synonyms=search_synonyms,
        )
    except Exception:
        return None

    user_prompt = format_user_prompt(raw_item)
    assistant_json = json.dumps(gt_item.model_dump(), ensure_ascii=False)

    return {
        "sku_id": raw_item.sku_id,
        "merchant_type": raw_item.merchant_type,
        "category": gt_item.category,
        "raw_input": raw_item.model_dump(),
        "ground_truth": gt_item.model_dump(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_json},
        ],
    }


def build_sft_dataset(
    raw_jsonl_path: str | Path = "data/raw/openfoodfacts_br_raw.jsonl",
    train_size: int = 800,
    val_size: int = 150,
    test_size: int = 200,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """
    Constrói os splits train, validation e test a partir do snapshot real
    do Open Food Facts Brasil, sem sobreposição de EAN ou prompt.
    """
    raw_path = Path(raw_jsonl_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Arquivo bruto real não encontrado em {raw_path}. Execute python -m src.raw_ingestion primeiro."
        )

    rng = random.Random(seed)
    valid_records: list[dict[str, Any]] = []
    seen_skus: set[str] = set()
    seen_prompts: set[str] = set()

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_prod = json.loads(line)
            transformed = transform_raw_off_record(raw_prod, rng)
            if transformed is None:
                continue
            sku_id = transformed["sku_id"]
            user_prompt = transformed["messages"][1]["content"]
            if sku_id in seen_skus or user_prompt in seen_prompts:
                continue
            seen_skus.add(sku_id)
            seen_prompts.add(user_prompt)
            valid_records.append(transformed)

    total_needed = train_size + val_size + test_size
    if len(valid_records) < total_needed:
        raise ValueError(
            f"Registros reais válidos insuficientes: encontrados {len(valid_records)}, "
            f"necessários {total_needed}."
        )

    rng.shuffle(valid_records)

    return {
        "train": valid_records[:train_size],
        "validation": valid_records[train_size : train_size + val_size],
        "test": valid_records[train_size + val_size : total_needed],
    }


def save_splits_to_jsonl(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Salva os splits reais processados em arquivos JSONL."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for split_name, rows in splits.items():
        file_path = out_path / f"{split_name}.jsonl"
        with file_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        saved_paths[split_name] = file_path

    return saved_paths


if __name__ == "__main__":
    splits = build_sft_dataset()
    paths = save_splits_to_jsonl(splits)
    all_rows = splits["train"] + splits["validation"] + splits["test"]
    cat_counts = Counter(r["category"] for r in all_rows)
    tag_counts = Counter(tag for r in all_rows for tag in r["ground_truth"]["dietary_tags"])

    print("=== Dataset Real SFT Construído com Sucesso ===")
    for name, path in paths.items():
        print(f" - Split '{name}': {len(splits[name])} SKUs reais -> {path}")
    print("\nDistribuição por Categoria Canônica (1.150 SKUs Reais):")
    for cat, cnt in cat_counts.most_common():
        print(f"   * {cat}: {cnt}")
    print("\nDistribuição de Dietary Tags Reais:")
    for tag, cnt in tag_counts.most_common():
        print(f"   * {tag}: {cnt}")
