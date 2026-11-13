from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OFF_GLOBAL_API_URL = "https://world.openfoodfacts.org/api/v2/search"
FIELDS = (
    "code,product_name,product_name_pt,generic_name_pt,brands,quantity,"
    "categories,categories_tags,labels_tags,allergens_tags,"
    "ingredients_analysis_tags,ingredients_text_pt,ingredients_text"
)

CATEGORY_PARTITIONS = [
    "pizzas",
    "sandwiches",
    "burgers",
    "meals",
    "meats",
    "poultry",
    "seafood",
    "fishes",
    "snacks",
    "salty-snacks",
    "beverages",
    "carbonated-drinks",
    "juices-and-nectars",
    "dairies",
    "cheeses",
    "yogurts",
    "desserts",
    "frozen-desserts",
    "biscuits-and-cakes",
    "breads",
    "chocolates",
    "plant-based-foods",
    "pastas",
    "sauces",
    "cereals-and-potatoes",
]


def is_valid_real_product(raw_prod: dict[str, Any]) -> bool:
    """Filtra apenas produtos reais com EAN, nome legível e ingredientes reais."""
    code = str(raw_prod.get("code") or "").strip()
    if not re.fullmatch(r"\d{8,14}", code):
        return False

    name = (raw_prod.get("product_name_pt") or raw_prod.get("product_name") or "").strip()
    if len(name) < 3:
        return False

    ingredients = (raw_prod.get("ingredients_text_pt") or raw_prod.get("ingredients_text") or "").strip()
    if len(ingredients) < 10:
        return False

    return True


def load_existing_snapshot(file_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Carrega produtos já salvos no disco para retomar sem perder progresso."""
    collected: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    if not file_path.exists():
        return collected, seen_codes

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                code = str(item.get("code") or "").strip()
                if code and code not in seen_codes and is_valid_real_product(item):
                    seen_codes.add(code)
                    collected.append(item)
            except json.JSONDecodeError:
                continue
    return collected, seen_codes


def save_snapshot(file_path: Path, items: list[dict[str, Any]]) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def fetch_openfoodfacts_br_products(
    target_count: int = 1250,
    page_size: int = 100,
    output_path: str | Path = "data/raw/openfoodfacts_br_raw.jsonl",
) -> Path:
    """
    Complementa o snapshot bruto de produtos reais comercializados no Brasil
    usando partições rasas por categoria para evitar rate-limit e HTTP 503.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    collected, seen_codes = load_existing_snapshot(out_file)
    print(f"Retomando snapshot: {len(collected)} SKUs reais já carregados do disco.")

    for category in CATEGORY_PARTITIONS:
        if len(collected) >= target_count:
            break

        for page in (1, 2):
            if len(collected) >= target_count:
                break

            url = (
                f"{OFF_GLOBAL_API_URL}?countries_tags_en=brazil"
                f"&categories_tags_en={category}&page_size={page_size}&page={page}&fields={FIELDS}"
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "iFoodCatalogSLMPortfolio/1.0 (Contact: portfolio-research-br)"},
            )

            payload = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    wait_sec = 3.0 * (attempt + 1)
                    print(f"  [Aviso] {category} p.{page} (tentativa {attempt + 1}/3): {exc}. Aguardando {wait_sec}s...")
                    time.sleep(wait_sec)

            if not payload:
                continue

            products = payload.get("products", [])
            if not products:
                break

            added = 0
            for prod in products:
                code = str(prod.get("code") or "").strip()
                if code in seen_codes:
                    continue
                if is_valid_real_product(prod):
                    seen_codes.add(code)
                    collected.append(prod)
                    added += 1
                    if len(collected) >= target_count:
                        break

            save_snapshot(out_file, collected)
            print(
                f"Categoria '{category}' (p.{page}): +{added:02d} novos SKUs "
                f"(Total acumulado: {len(collected)}/{target_count})"
            )
            time.sleep(1.8)

    print(f"\nSnapshot bruto finalizado com sucesso: {len(collected)} SKUs reais em {out_file}")
    return out_file


if __name__ == "__main__":
    fetch_openfoodfacts_br_products()
