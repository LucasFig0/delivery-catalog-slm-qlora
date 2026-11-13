import pytest
from pydantic import ValidationError

from src.schemas import EnrichedCatalogItem, RawCatalogItem, parse_slm_json_output


def test_valid_enriched_catalog_item():
    item = EnrichedCatalogItem(
        normalized_title="Combo Burger Artesanal Bacon 180g com Fritas e Coca-Cola Zero 350ml",
        category="Lanches",
        main_protein_or_base="Bovino",
        portion_size="Individual",
        dietary_tags=["Zero Açúcar"],
        is_combo=True,
        volume_or_weight="180g",
        components=["Hambúrguer bovino 180g", "Queijo cheddar", "Bacon", "Batata frita", "Coca-Cola Zero 350ml"],
        search_synonyms=["hamburguer artesanal", "combo burger", "lanche com fritas"],
    )
    assert item.is_combo is True
    assert item.volume_or_weight == "180g"
    assert item.dietary_tags == ["Zero Açúcar"]


def test_measure_normalization_variants():
    item_ml = EnrichedCatalogItem(
        normalized_title="Refrigerante Guaraná Antarctica Zero 1.5L",
        category="Bebidas",
        main_protein_or_base="Bebida/Outro",
        portion_size="Família (3+ Pessoas)",
        dietary_tags=["Zero Açúcar"],
        is_combo=False,
        volume_or_weight="1,5 l",
        components=["Guaraná Antarctica Zero 1.5L"],
        search_synonyms=["guarana zero", "refrigerante 1.5 litros"],
    )
    assert item_ml.volume_or_weight == "1.5L"


def test_food_safety_guardrail_blocks_vegan_meat():
    with pytest.raises(ValidationError) as exc_info:
        EnrichedCatalogItem(
            normalized_title="Picanha na Brasa com Macaxeira Frita",
            category="Brasileira",
            main_protein_or_base="Bovino",
            portion_size="Casal (2 Pessoas)",
            dietary_tags=["Vegano"],
            is_combo=False,
            volume_or_weight="500g",
            components=["Picanha bovina", "Macaxeira frita"],
            search_synonyms=["churrasco", "carne com mandioca"],
        )
    assert "Violação de segurança alimentar" in str(exc_info.value)


def test_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        EnrichedCatalogItem(
            normalized_title="Açaí Tradicional 500ml",
            category="Açaí e Sorvetes",
            main_protein_or_base="Açaí/Frutas",
            portion_size="Individual",
            dietary_tags=[],
            is_combo=False,
            volume_or_weight="500ml",
            components=["Polpa de açaí"],
            search_synonyms=["acai na tigela", "sobremesa gelada"],
            hallucinated_field="invalid",
        )


def test_parse_slm_json_output_clean_and_fenced():
    clean_json = (
        '{"normalized_title": "Temaki Salmão Completo", "category": "Japonesa", '
        '"main_protein_or_base": "Peixe e Frutos do Mar", "portion_size": "Individual", '
        '"dietary_tags": [], "is_combo": false, "volume_or_weight": "150g", '
        '"components": ["Salmão fresco", "Arroz japonês", "Alga nori", "Cream cheese"], '
        '"search_synonyms": ["comida japonesa", "cone de salmao", "sushi"]}'
    )
    parsed, err, repaired = parse_slm_json_output(clean_json)
    assert err is None
    assert repaired is False
    assert parsed is not None
    assert parsed.category == "Japonesa"

    fence = chr(96) * 3
    fenced_output = f"Aqui está o JSON solicitado:\n{fence}json\n{clean_json}\n{fence}"
    parsed_fenced, err_fenced, repaired_fenced = parse_slm_json_output(fenced_output)
    assert err_fenced is None
    assert repaired_fenced is True
    assert parsed_fenced is not None
    assert parsed_fenced.normalized_title == "Temaki Salmão Completo"


def test_raw_catalog_item_validation():
    raw = RawCatalogItem(
        sku_id="SKU-REC-001",
        raw_title="Comb 2x Burg 180g + Fritas",
        raw_description="Pão brioche, blend 180g, cheddar",
        merchant_type="Hamburgueria",
        price_brl=49.90,
    )
    assert raw.sku_id == "SKU-REC-001"
