import pytest

from src.qlora_config import (
    QWEN_LINEAR_TARGET_MODULES,
    QLoRAHyperparameters,
    estimate_qlora_memory_and_params,
)


def test_default_qlora_hyperparameters_and_invariants():
    cfg = QLoRAHyperparameters()
    cfg.validate_invariants()
    assert cfg.base_model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.effective_train_batch_size == 16
    assert cfg.lora_scaling_factor == 2.0
    assert set(cfg.target_modules) == set(QWEN_LINEAR_TARGET_MODULES)


def test_invalid_hyperparameters_raise_error():
    bad_cfg = QLoRAHyperparameters(lora_r=32, lora_alpha=16)
    with pytest.raises(ValueError):
        bad_cfg.validate_invariants()


def test_estimate_qlora_memory_and_trainable_params():
    stats = estimate_qlora_memory_and_params()
    assert 15_000_000 <= stats["trainable_lora_params"] <= 22_000_000
    assert 1.0 <= stats["trainable_params_pct"] <= 1.5
    assert stats["total_estimated_vram_gb"] < 6.0
    assert stats["fits_single_t4_16gb"] == 1.0


def test_export_kwargs_formats():
    cfg = QLoRAHyperparameters()
    bnb = cfg.to_bnb_kwargs()
    lora = cfg.to_lora_kwargs()
    sft = cfg.to_sft_kwargs(output_dir="tmp_out")
    assert bnb["bnb_4bit_quant_type"] == "nf4"
    assert lora["r"] == 16
    assert sft["output_dir"] == "tmp_out"
    assert sft["optim"] == "paged_adamw_8bit"
