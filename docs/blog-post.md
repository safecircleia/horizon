# Introducing Horizon: Child Safety Detection That Never Leaves the Phone

Every parent faces the same impossible choice: monitor your child's messages and invade their privacy, or trust blindly and hope for the best. Existing safety tools force this tradeoff — they either upload private conversations to remote servers for analysis, or rely on simple keyword filters that miss nuanced threats and flag innocent messages.

Today we're introducing **Horizon**, SafeCircle's on-device AI model that changes this equation entirely. It detects real safety risks — grooming, bullying, threats, and more — in under 100 milliseconds, directly on your child's phone. No messages are ever sent to our servers. No one reads their conversations. Not even us.

## The Problem With Current Solutions

Server-side content moderation works by sending your child's messages to a company's cloud infrastructure, where AI models scan them for risks. This approach has two fundamental flaws:

1. **Privacy violation by design** — every message passes through a third party's servers, creating a permanent record of your child's private conversations.
2. **Single point of failure** — if the service goes down, monitoring stops entirely.

Keyword-based filtering is the alternative, but it's trivially circumvented ("meet me at the p4rk") and generates endless false positives from normal teenage conversation.

## How Horizon Works

Horizon takes a fundamentally different approach. Instead of sending data to the cloud, we bring the AI to the device.

The system uses two models working in tandem:

**Horizon-Full** is our high-capacity teacher model — a 3 billion parameter language model fine-tuned specifically for risk detection. It understands context, subtext, and the subtle patterns that characterize real threats. This model runs on our servers during training only.

**Horizon-Mobile** is a tiny 25MB classifier that runs entirely on-device. We trained it using *knowledge distillation* — a technique where a large, accurate model teaches a smaller model to replicate its decisions. The result is a mobile model that captures the intelligence of the larger system in a package small enough to run on any modern smartphone.

## What It Detects

Horizon identifies seven categories of risk, each trained on thousands of realistic conversation examples:

- **Grooming** — incremental boundary violations, secrecy demands, trust manipulation
- **Bullying** — repeated intimidation, social exclusion, humiliation
- **Sexual content** — explicit or suggestive material directed at minors
- **Isolation** — attempts to separate a child from their support network
- **Personal information** — solicitation of addresses, schools, photos
- **Platform migration** — requests to move to less-monitored channels
- **Threats** — direct or implied physical or psychological harm

For each detection, Horizon provides a severity rating (none through critical), a confidence score, and — critically — an explanation of *which specific words or phrases* triggered the alert.

## Privacy by Architecture

This isn't privacy by policy. It's privacy by architecture. The 25MB model file lives on the device. Inference happens on the device. Message content never touches a network connection.

There's no server to breach. No database of children's messages to leak. No employee who could access private conversations. The mathematical impossibility of extracting training data from a neural network means that even the model file itself reveals nothing about any individual child's messages.

## The Technical Innovation

Building a model this small without sacrificing accuracy required solving several hard problems:

**Knowledge distillation** — We don't just compress the model; we teach the small model to replicate the *reasoning patterns* of the large model. When the teacher model says "this is 70% grooming, 20% isolation, 10% benign," that uncertainty itself is valuable training signal.

**Dual-head classification** — A single forward pass through the model simultaneously classifies both the risk category and severity level, keeping inference under our 100ms target.

**Explainability** — Using Integrated Gradients, we compute which specific tokens in the conversation most influenced the model's decision. Parents see *why* something was flagged, not just that it was.

## What's Next

Horizon is currently trained on English-language conversations. We're actively expanding to Spanish, Portuguese, and French. We're also working on temporal pattern detection — identifying risk that emerges across multiple conversations over days or weeks, not just within a single message.

The full technical paper, including mathematical formulations, architecture details, and training methodology, is available for download below.

---

**[Download the full technical paper (PDF)](#)**

---

*Horizon is open-source. View the code at [github.com/safecircleia/horizon](https://github.com/safecircleia/horizon).*
