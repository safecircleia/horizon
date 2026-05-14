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


def _configure_attention(attn_impl: str) -> str:
    """Configure the attention backend and return the transformers attn_implementation value."""
    if attn_impl == "cudnn_attention":
        if not torch.backends.cuda.cudnn_sdp_enabled():
            raise RuntimeError(
                "attn_implementation=cudnn_attention requires PyTorch 2.5+ built with cuDNN SDP support."
            )
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(True)
        return "sdpa"
    return attn_impl


def load_model_and_tokenizer(config: dict) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and tokenizer, applying QLoRA and attention config."""
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

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=TORCH_DTYPE_MAP[quant_cfg["bnb_4bit_compute_dtype"]],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=model_dtype,
        )
        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        attn_impl = _configure_attention(model_cfg.get("attn_implementation", "sdpa"))
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            device_map="cuda:0",
            trust_remote_code=True,
            torch_dtype=model_dtype,
            attn_implementation=attn_impl,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=model_dtype,
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
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.set_adapter("default")

    return model, tokenizer
