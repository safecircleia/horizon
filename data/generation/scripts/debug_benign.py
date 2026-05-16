"""One-shot debug script — run from project root: python -m data.generation.scripts.debug_benign"""
import asyncio, uuid, datetime, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.generation.scripts.generate import create_generator, load_config
from data.generation.prompts.base import create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel, Message, ConversationLabel, SyntheticConversation
from data.generation.validators.quality import validate_conversation_quality

async def main():
    config = load_config("data/generation/config.yaml")
    quality_config = config.get("quality", {})
    gen = create_generator("vllm", config)
    prompt = create_conversation_prompt(RiskCategory.BENIGN, RiskLevel.NONE, 15, 8)
    result = await gen.generate(prompt)

    print("=== HTTP result ===")
    print("success:", result.success, "| error:", result.error)
    print()

    print("=== Parsed conversation keys ===")
    print(list(result.conversation.keys()) if result.conversation else "None")
    print()

    raw_messages = result.conversation.get("messages", []) if result.conversation else []
    print(f"=== Messages ({len(raw_messages)}) ===")
    for i, m in enumerate(raw_messages[:3]):
        print(f"  [{i}] {m}")
    print()

    print("=== Parsing Messages ===")
    try:
        messages = [Message(**m) for m in raw_messages]
        print(f"  OK — {len(messages)} messages parsed")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    print("=== Quality validation ===")
    is_valid, errors = validate_conversation_quality(
        messages,
        min_length=quality_config.get("min_conversation_length", 4),
        max_length=quality_config.get("max_conversation_length", 30),
        min_unique_tokens=quality_config.get("min_unique_tokens", 20),
        allow_consecutive_roles=True,
    )
    print(f"  valid={is_valid} errors={errors}")
    if not is_valid:
        return

    print("=== Building label ===")
    try:
        label = ConversationLabel(
            risk_level=RiskLevel.NONE,
            categories=[RiskCategory.BENIGN],
            severity_score=0.0,
            reasoning=result.conversation.get("reasoning", "Generated conversation"),
        )
        print(f"  OK — reasoning='{label.reasoning[:60]}'")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    print("=== Building SyntheticConversation ===")
    try:
        conv = SyntheticConversation(
            conversation_id=str(uuid.uuid4()),
            category=RiskCategory.BENIGN,
            messages=messages,
            label=label,
            metadata={"generator": "vllm", "child_age": 15, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "model": gen.model, "attempt": 1},
        )
        print(f"  OK — {len(conv.messages)} messages")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    print("\n=== ALL STEPS PASSED — benign generation should work ===")

async def concurrent():
    """Test 20 benign requests concurrently to reproduce the batch failure."""
    config = load_config("data/generation/config.yaml")
    gen = create_generator("vllm", config)
    quality_config = config.get("quality", {})

    async def one(i):
        from data.generation.validators.schemas import RiskCategory, RiskLevel, Message, ConversationLabel, SyntheticConversation
        prompt = create_conversation_prompt(RiskCategory.BENIGN, RiskLevel.NONE, 15, 8)
        result = await gen.generate(prompt)
        if not result.success:
            return f"[{i}] HTTP fail: {result.error}"
        try:
            raw = result.conversation.get("messages", [])
            messages = [Message(**m) for m in raw]
            is_valid, errors = validate_conversation_quality(
                messages,
                min_length=quality_config.get("min_conversation_length", 4),
                max_length=quality_config.get("max_conversation_length", 30),
                min_unique_tokens=quality_config.get("min_unique_tokens", 20),
                allow_consecutive_roles=True,
            )
            if not is_valid:
                return f"[{i}] quality fail: {errors}"
            label = ConversationLabel(
                risk_level=RiskLevel.NONE,
                categories=[RiskCategory.BENIGN],
                severity_score=0.0,
                reasoning=result.conversation.get("reasoning", "Generated conversation"),
            )
            return f"[{i}] OK"
        except Exception as e:
            return f"[{i}] EXCEPTION: {type(e).__name__}: {e}"

    results = await asyncio.gather(*[one(i) for i in range(20)])
    print("\n=== 20 concurrent benign requests ===")
    for r in results:
        print(" ", r)
    ok = sum(1 for r in results if "OK" in r)
    print(f"\n{ok}/20 succeeded")

asyncio.run(concurrent())
