"""Mobile model definition for LiteRT-LM deployment.

The mobile model is Gemma 3 1B fine-tuned with QLoRA on SafeCircle data.
This module holds constants and the system prompt shared by training,
export, and inference code.
"""

RISK_CATEGORIES = [
    "grooming",
    "bullying",
    "sexual_content",
    "isolation",
    "personal_info",
    "platform_migration",
    "threats",
    "benign",
]

RISK_LEVELS = ["none", "low", "medium", "high", "critical"]

MOBILE_BASE_MODEL = "google/gemma-3-1b-it"

SYSTEM_PROMPT = (
    "You are SafeCircle, a child safety AI. Analyse the conversation and respond "
    "with a JSON object only — no prose. Schema: "
    '{\"risk_level\": \"none|low|medium|high|critical\", '
    '\"categories\": [\"grooming\"|\"bullying\"|\"sexual_content\"|\"isolation\"|'
    '\"personal_info\"|\"platform_migration\"|\"threats\"|\"benign\"], '
    '\"confidence\": 0.0-1.0}'
)
