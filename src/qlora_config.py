from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

QWEN_LINEAR_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class QLoRAHyperparameters:
    """Especificação imutável e auditável do pipeline QLoRA para GPU NVIDIA T4 (16GB VRAM)."""

    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_dataset_id: str = "brazilian-delivery-catalog-sft"
    hf_adapter_repo_id: str = "qwen2.5-1.5b-ifood-catalog-qlora"

    # Quantização 4-bit NormalFloat (bitsandbytes)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"

    # Hiperparâmetros PEFT LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = field(default_factory=lambda: QWEN_LINEAR_TARGET_MODULES)

    # Hiperparâmetros SFTTrainer (TRL)
    max_seq_length: int = 768
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    num_train_epochs: int = 2
    optim: str = "paged_adamw_8bit"
    fp16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 25
    save_steps: int = 50
    seed: int = 42

    def validate_invariants(self) -> None:
        if self.bnb_4bit_quant_type not in {"nf4", "fp4"}:
            raise ValueError("bnb_4bit_quant_type deve ser 'nf4' ou 'fp4'.")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_r e lora_alpha devem ser inteiros positivos.")
        if self.lora_alpha < self.lora_r:
            raise ValueError("Recomenda-se lora_alpha >= lora_r para estabilidade de escala.")
        if len(self.target_modules) < 4:
            raise ValueError("target_modules deve cobrir pelo menos as projeções de atenção.")
        if self.effective_train_batch_size < 8:
            raise ValueError("Batch efetivo deve ser >= 8 para estabilidade do gradiente em SFT.")

    @property
    def effective_train_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def lora_scaling_factor(self) -> float:
        return self.lora_alpha / float(self.lora_r)

    def to_bnb_kwargs(self) -> dict[str, Any]:
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
        }

    def to_lora_kwargs(self) -> dict[str, Any]:
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": list(self.target_modules),
        }

    def to_sft_kwargs(self, output_dir: str = "artifacts/lora_checkpoints") -> dict[str, Any]:
        return {
            "output_dir": output_dir,
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "num_train_epochs": self.num_train_epochs,
            "optim": self.optim,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        data["effective_train_batch_size"] = self.effective_train_batch_size
        data["lora_scaling_factor"] = self.lora_scaling_factor
        return data


def estimate_qlora_memory_and_params(
    config: QLoRAHyperparameters | None = None,
    num_layers: int = 28,
    hidden_size: int = 1536,
    intermediate_size: int = 8960,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    total_base_params: int = 1_543_714_304,
) -> dict[str, float]:
    """
    Calcula analiticamente o número de parâmetros treináveis do adaptador LoRA
    na arquitetura Qwen2.5-1.5B e a estimativa de consumo de VRAM em GPU T4.
    """
    cfg = config or QLoRAHyperparameters()
    cfg.validate_invariants()
    r = cfg.lora_r
    kv_dim = num_kv_heads * head_dim

    # Parâmetros (A + B) por camada Transformer para cada projeção linear
    q_params = (hidden_size * r) + (r * hidden_size)
    k_params = (hidden_size * r) + (r * kv_dim)
    v_params = (hidden_size * r) + (r * kv_dim)
    o_params = (hidden_size * r) + (r * hidden_size)

    gate_params = (hidden_size * r) + (r * intermediate_size)
    up_params = (hidden_size * r) + (r * intermediate_size)
    down_params = (intermediate_size * r) + (r * hidden_size)

    per_layer_lora_params = (
        q_params + k_params + v_params + o_params + gate_params + up_params + down_params
    )
    trainable_params = num_layers * per_layer_lora_params
    trainable_pct = (trainable_params / float(total_base_params)) * 100.0

    # Estimativa de VRAM (GB)
    # 1. Pesos base quantizados em 4.5 bits efetivos (NF4 + double quant overhead)
    base_weights_gb = (total_base_params * 0.5625) / (1024**3)
    # 2. Pesos LoRA FP16 + Gradientes FP16 + Estados Paged AdamW 8-bit (~6 bytes/param treinável)
    lora_and_optim_gb = (trainable_params * 6.0) / (1024**3)
    # 3. Ativações com gradient checkpointing + contexto CUDA na T4
    activations_and_cuda_gb = 2.45

    total_estimated_vram_gb = base_weights_gb + lora_and_optim_gb + activations_and_cuda_gb

    return {
        "total_base_params": float(total_base_params),
        "trainable_lora_params": float(trainable_params),
        "trainable_params_pct": round(trainable_pct, 4),
        "base_weights_vram_gb": round(base_weights_gb, 3),
        "lora_and_optim_vram_gb": round(lora_and_optim_gb, 3),
        "total_estimated_vram_gb": round(total_estimated_vram_gb, 3),
        "fits_single_t4_16gb": 1.0 if total_estimated_vram_gb < 14.5 else 0.0,
    }
