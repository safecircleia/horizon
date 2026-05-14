"""Model and tokenizer loading for QLoRA training."""

import os
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_te_sdpa_patched = False


def _patch_sdpa_with_te_cudnn(num_heads: int, head_dim: int, dtype: torch.dtype) -> None:
    """Replace F.scaled_dot_product_attention with TE cuDNN FusedAttention.

    TE's DotProductAttention expects (seq, batch, heads, dim) — i.e. qkv_format="bshd"
    with tensors shaped [b, s, h, d]. Transformers emits [b, h, s, d] (bhsd), so we
    transpose in and out. NVTE_FLASH_ATTN=0 forces the cuDNN sub-backend (no flash-attn).
    """
    global _te_sdpa_patched
    if _te_sdpa_patched:
        return

    try:
        import transformer_engine.pytorch as te
    except ImportError:
        print("Warning: transformer-engine not installed; falling back to sdpa. "
              "Install: pip install transformer-engine")
        return

    # Disable flash-attn inside TE so it always routes to cuDNN FusedAttention
    os.environ.setdefault("NVTE_FLASH_ATTN", "0")

    dpa = te.DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=head_dim,
        attention_dropout=0.0,
        attn_mask_type="causal",
        qkv_format="bshd",
    ).to(dtype=dtype, device="cuda")

    _orig_sdpa = F.scaled_dot_product_attention

    def _te_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        # transformers passes [b, h, s, d] — transpose to [b, s, h, d] for TE
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        out = dpa(q, k, v)
        # transpose back to [b, h, s, d]
        return out.transpose(1, 2).contiguous()

    F.scaled_dot_product_attention = _te_sdpa
    _te_sdpa_patched = True
    print("TE cuDNN FusedAttention active (NVTE_FLASH_ATTN=0, sub-backend 1).")


def load_model_and_tokenizer(
    config: dict,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model with QLoRA config ready for training."""
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    quant_cfg = config["quantization"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_dtype = TORCH_DTYPE_MAP[model_cfg["torch_dtype"]]
    use_4bit = quant_cfg.get("load_in_4bit", True)
    attn_impl = model_cfg.get("attn_implementation")

    if use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        compute_dtype = TORCH_DTYPE_MAP[quant_cfg["bnb_4bit_compute_dtype"]]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
            llm_int8_enable_fp32_cpu_offload=quant_cfg.get("llm_int8_enable_fp32_cpu_offload", False),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=model_dtype,
            max_memory=quant_cfg.get("max_memory", None),
        )
        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        load_kwargs = dict(
            device_map="cuda:0",
            trust_remote_code=True,
            dtype=model_dtype,
            # Always load with sdpa; TE patches F.scaled_dot_product_attention directly
            attn_implementation="sdpa",
        )
        model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"], **load_kwargs)

        if attn_impl == "cudnn_attention":
            num_heads = model.config.num_attention_heads
            head_dim = model.config.hidden_size // num_heads
            _patch_sdpa_with_te_cudnn(num_heads, head_dim, model_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            device_map="cpu",
            trust_remote_code=True,
            dtype=model_dtype,
        )

    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_for_inference(
    checkpoint_path: str,
    base_model: Optional[str] = None,
) -> Tuple[PeftModel, AutoTokenizer]:
    """Load a trained LoRA checkpoint for inference."""
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    base = base_model or peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.set_adapter("default")

    return model, tokenizer
