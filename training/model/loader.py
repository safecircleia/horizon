"""Model and tokenizer loading for QLoRA training."""

from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


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
        # Full precision GPU path (L4 / high-VRAM GPUs)
        kwargs = dict(
            device_map="cuda:0",
            trust_remote_code=True,
            dtype=model_dtype,
        )
        attn_impl = model_cfg.get("attn_implementation")
        if attn_impl in ("flash_attention_4", "flash_attention_2"):
            try:
                import flash_attn  # noqa: F401
                import importlib.metadata
                try:
                    fa_ver = importlib.metadata.version("flash-attn")
                except importlib.metadata.PackageNotFoundError:
                    fa_ver = "unknown"
                # FA4 (flash-attn-4) uses flash_attn.cute and doesn't integrate with
                # transformers' attn_implementation — FA2 is the correct transformers hook.
                # FA4 kernels are used automatically on Hopper when flash-attn-4 is installed.
                kwargs["attn_implementation"] = "flash_attention_2"
                print(f"Using Flash Attention 2 via transformers (flash-attn {fa_ver})")
            except ImportError:
                print("Warning: flash-attn not installed, falling back to sdpa. "
                      "Install: pip install flash-attn --no-build-isolation")
                kwargs["attn_implementation"] = "sdpa"
        elif attn_impl:
            kwargs["attn_implementation"] = attn_impl
        model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"], **kwargs)
    else:
        # CPU fallback
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
