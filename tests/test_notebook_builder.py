import json
from pathlib import Path

from scripts.build_kaggle_notebook import build_kaggle_notebook


def test_build_kaggle_notebook_structure(tmp_path: Path):
    nb_path = build_kaggle_notebook(output_path=tmp_path / "test_nb.ipynb")
    assert nb_path.exists()

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 9

    full_source = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "LucasFig0/brazilian-delivery-catalog-sft" in full_source
    assert "LucasFig0/qwen2.5-1.5b-ifood-catalog-qlora" in full_source
    assert "Qwen/Qwen2.5-1.5B-Instruct" in full_source
    assert "tokenize_with_completion_mask" in full_source


def test_all_notebook_code_cells_compile_cleanly(tmp_path: Path):
    nb_path = build_kaggle_notebook(output_path=tmp_path / "compile_nb.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        python_lines = [
            line for line in cell["source"] if not line.lstrip().startswith("!")
        ]
        code_str = "".join(python_lines)
        compile(code_str, f"<notebook_cell_{idx + 1}>", "exec")
