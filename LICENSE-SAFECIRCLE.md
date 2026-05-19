# SafeCircle Research License (SRL-1.0)

**Version 1.0 — May 2026**
**Copyright © 2026 SafeCircle. All rights reserved.**

---

## 1. Definitions

- **"Model"** means the machine learning model weights, configuration files, tokenizer files, and associated code released under this license.
- **"Derivative Work"** means any work that incorporates, fine-tunes, distills, quantizes, or is otherwise derived from the Model.
- **"You"** (or **"Your"**) means the individual or legal entity exercising rights under this license.
- **"Commercial Use"** means any use of the Model, in whole or in part, for direct or indirect commercial advantage or monetary compensation.
- **"SafeCircle"** means SafeCircle Ltd. and its authorized representatives.

---

## 2. Grant of License

Subject to the terms and conditions of this license, SafeCircle grants You a non-exclusive, non-transferable, worldwide, royalty-free license to:

- Use the Model for **research, education, and non-commercial purposes**.
- Reproduce and share the Model for non-commercial research purposes, provided attribution requirements in Section 4 are met.
- Create Derivative Works for non-commercial research purposes, subject to Section 5.

---

## 3. Prohibited Uses

The following uses are **strictly prohibited**, regardless of intent:

### 3.1 Harm to Minors
You may **not** use the Model, or any Derivative Work, to:
- Generate, facilitate, normalize, or distribute content that sexually exploits, grooms, threatens, or otherwise harms minors.
- Build tools, systems, or pipelines designed to produce harmful content targeting children.
- Train or fine-tune models on outputs derived from this Model for any purpose that involves child exploitation or abuse.

### 3.2 Safety System Evasion
You may **not** use the Model to:
- Develop, train, or assist in developing systems that evade, deceive, or circumvent child safety detection systems.
- Probe, red-team, or reverse-engineer child safety classifiers for purposes other than legitimate academic research with institutional ethics approval.
- Generate adversarial examples intended to fool child safety monitoring systems in production environments.

### 3.3 Commercial Use
You may **not** use the Model for Commercial Use without **prior written permission** from SafeCircle. To request a commercial license, contact: **legal@safecircle.tech**

### 3.4 Redistribution Without Attribution
You may **not** redistribute the Model or any Derivative Work without:
- Clearly crediting SafeCircle as the original creator.
- Including a link to the original model repository.
- Including a copy of this license.
- Clearly marking any modifications made to the original Model.

---

## 4. Attribution Requirements

Any publication, product, or service that uses this Model must include the following attribution:

> *"This work uses Horizon, developed by SafeCircle (https://huggingface.co/safecircleai/horizon-full)."*

Academic publications must cite:
```bibtex
@misc{horizon2026,
  title={Horizon: Child Safety Risk Detection via Fine-tuned LLMs},
  author={SafeCircle},
  year={2026},
  url={https://huggingface.co/safecircleai/horizon-full}
}
```

---

## 5. Derivative Works

If You create and distribute a Derivative Work:
- You must release it under this same license (SRL-1.0) or a more restrictive license.
- You must clearly indicate that the work is a Derivative Work of the SafeCircle Horizon model.
- You must not imply that SafeCircle endorses or is affiliated with your Derivative Work.
- All prohibited uses in Section 3 apply equally to Derivative Works.

---

## 6. Responsible Use

This Model was developed for child safety research and protective applications. Users are encouraged to:
- Deploy the Model only as an assistive tool, not as a sole decision-making system.
- Ensure human review of all model outputs in production settings.
- Report misuse, vulnerabilities, or unexpected behavior to **legal@safecircle.tech**
- Follow applicable laws regarding child protection in your jurisdiction.

---

## 7. Disclaimer of Warranties

THE MODEL IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. SAFECIRCLE DOES NOT WARRANT THAT THE MODEL IS FREE OF ERRORS, THAT OUTPUTS ARE ACCURATE, OR THAT IT WILL MEET YOUR REQUIREMENTS.

---

## 8. Limitation of Liability

IN NO EVENT SHALL SAFECIRCLE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING FROM THE USE OR INABILITY TO USE THE MODEL, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

---

## 9. Termination

This license terminates automatically if You fail to comply with any of its terms. Upon termination, You must cease all use and distribution of the Model and destroy all copies in Your possession.

---

## 10. Contact

For commercial licensing, partnership inquiries, or to report misuse:

**Email:** legal@safecircle.tech
**Website:** https://safecircle.tech
