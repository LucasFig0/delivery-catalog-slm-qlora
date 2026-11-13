from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CatalogCategory = Literal[
    "Lanches",
    "Pizzas",
    "Brasileira",
    "Japonesa",
    "Açaí e Sorvetes",
    "Bebidas",
    "Sobremesas",
    "Saudável e Fit",
    "Padaria e Cafeteria",
    "Mercado",
]

MainProteinOrBase = Literal[
    "Bovino",
    "Frango",
    "Suíno",
    "Peixe e Frutos do Mar",
    "Vegetariano/Vegano",
    "Massa/Queijo",
    "Açaí/Frutas",
    "Bebida/Outro",
]

PortionSize = Literal[
    "Individual",
    "Casal (2 Pessoas)",
    "Família (3+ Pessoas)",
    "Porção/Acompanhamento",
    "Unitário",
]

DietaryTag = Literal[
    "Zero Açúcar",
    "Sem Lactose",
    "Sem Glúten",
    "Vegano",
    "Vegetariano",
    "Low Carb/Fit",
]

ANIMAL_PROTEINS = {"Bovino", "Frango", "Suíno", "Peixe e Frutos do Mar"}


class RawCatalogItem(BaseModel):
    """Contrato de entrada representando um SKU bruto vindo de restaurante parceiro."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(..., min_length=3, max_length=64)
    raw_title: str = Field(..., min_length=2, max_length=200)
    raw_description: str = Field(default="", max_length=600)
    merchant_type: str = Field(..., min_length=2, max_length=80)
    price_brl: float = Field(..., gt=0.0)


class EnrichedCatalogItem(BaseModel):
    """Contrato estrito de saída do SLM para indexação e busca no catálogo."""

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

    @field_validator("normalized_title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if len(cleaned) < 3:
            raise ValueError("normalized_title deve conter pelo menos 3 caracteres úteis.")
        return cleaned

    @field_validator("volume_or_weight")
    @classmethod
    def normalize_measure(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in {"null", "none", "n/a", "-"}:
            return None
        cleaned = cleaned.replace(",", ".")
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ml|l|g|kg|un)", cleaned, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Formato de medida inválido: '{value}'. Use padrões como 350ml, 180g, 1.5L.")
        amount, unit = match.group(1), match.group(2).lower()
        if unit == "l":
            unit = "L"
        return f"{amount}{unit}"

    @field_validator("dietary_tags")
    @classmethod
    def deduplicate_tags(cls, value: list[DietaryTag]) -> list[DietaryTag]:
        seen: set[str] = set()
        result: list[DietaryTag] = []
        for tag in value:
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result

    @field_validator("components", "search_synonyms")
    @classmethod
    def clean_string_lists(cls, value: list[str]) -> list[str]:
        cleaned_list: list[str] = []
        for item in value:
            cleaned = re.sub(r"\s+", " ", item).strip()
            if cleaned and cleaned not in cleaned_list:
                cleaned_list.append(cleaned)
        if not cleaned_list:
            raise ValueError("A lista não pode estar vazia após higienização.")
        return cleaned_list

    @model_validator(mode="after")
    def validate_food_safety_consistency(self) -> "EnrichedCatalogItem":
        if self.main_protein_or_base in ANIMAL_PROTEINS:
            forbidden = {"Vegano", "Vegetariano"}.intersection(set(self.dietary_tags))
            if forbidden:
                raise ValueError(
                    f"Violação de segurança alimentar: proteína '{self.main_protein_or_base}' "
                    f"é incompatível com as tags {sorted(forbidden)}."
                )
        return self


def parse_slm_json_output(
    raw_text: str,
) -> tuple[Optional[EnrichedCatalogItem], Optional[str], bool]:
    """
    Analisa a saída textual do SLM e valida contra EnrichedCatalogItem.
    Retorna: (item_validado, mensagem_de_erro, precisou_reparo_de_texto).
    """
    if not raw_text or not raw_text.strip():
        return None, "Empty output from model", False

    text = raw_text.strip()
    repaired = False
    fence = chr(96) * 3

    if fence in text:
        repaired = True
        text = re.sub(rf"^{fence}(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(rf"\s*{fence}$", "", text)
        text = text.strip()

    if not (text.startswith("{") and text.endswith("}")):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            repaired = True
            text = text[start : end + 1]
        else:
            return None, "No JSON object boundaries found in output", repaired

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}", repaired

    if not isinstance(payload, dict):
        return None, "Parsed JSON root is not an object/dict", repaired

    try:
        validated = EnrichedCatalogItem.model_validate(payload)
        return validated, None, repaired
    except Exception as exc:
        return None, f"SchemaValidationError: {exc}", repaired
