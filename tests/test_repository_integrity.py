import re
from pathlib import Path


def test_required_deliverables_exist_and_non_empty():
    required_files = [
        "Dockerfile",
        "README.md",
        "docs/TECHNICAL_ARTICLE.md",
        "requirements.txt",
        "src/schemas.py",
        "src/raw_ingestion.py",
        "src/dataset_builder.py",
        "src/evaluator.py",
        "src/qlora_config.py",
        "notebooks/ifood_catalog_qlora_benchmark.ipynb",
        "artifacts/metrics/benchmark_results.json",
        "data/raw/openfoodfacts_br_raw.jsonl",
    ]
    for rel_path in required_files:
        p = Path(rel_path)
        assert p.exists(), f"Entregável obrigatório ausente: {rel_path}"
        assert p.stat().st_size > 100, f"Arquivo {rel_path} está vazio ou incompleto."


def test_readme_and_article_contain_real_benchmark_links_and_metrics():
    readme = Path("README.md").read_text(encoding="utf-8")
    article = Path("docs/TECHNICAL_ARTICLE.md").read_text(encoding="utf-8")

    for content in (readme, article):
        assert "LucasFig0/brazilian-delivery-catalog-sft" in content
        assert "75.00%" in content or "75,00%" in content
        assert "18.50%" in content or "18,50%" in content
        assert "EAN-13" in content


def test_dockerfile_follows_security_best_practices():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "USER appuser" in dockerfile
    assert "pytest" in dockerfile


def test_no_hardcoded_hf_tokens_in_tracked_files():
    token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    checked_dirs = ["src", "tests", "scripts", "docs", "notebooks", "artifacts"]
    root_files = ["README.md", "Dockerfile", "requirements.txt", ".gitignore"]

    files_to_scan = [Path(f) for f in root_files if Path(f).exists()]
    for d in checked_dirs:
        dir_path = Path(d)
        if dir_path.exists():
            files_to_scan.extend(p for p in dir_path.rglob("*") if p.is_file() and p.suffix != ".pyc")

    for file_path in files_to_scan:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        match = token_pattern.search(text)
        assert match is None, f"Vazamento de token detectado em {file_path}!"
