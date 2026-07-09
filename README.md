# You're an Open Book (Weights Included)

**Reading an agent's intent from open-weight model activations, before the tool call.**

Paper: [`the-tell.pdf`](the-tell.pdf)

---

## The one-line version

When an AI agent is about to do something, its intent is already sitting in the model's activations,
a moment before it acts. If you run the model yourself, you can read that intent with a single number,
before the agent has written a word of its reply.

On **Qwen2.5-1.5B** (a small open model that runs on a laptop, no GPU), a linear read of one middle
layer tells a malicious agent request from a benign one at **AUC 0.993**, held out by template, at
**zero false alarms for 95% of attacks caught**. It holds across models and families: **Qwen2.5-3B
1.000, Mistral-7B 0.997**. The cost is one dot product against a stored direction we call an **intent
probe** (a 16 KB file), which adds **under 1%** to a forward pass.

### What we also measured
- **Deception:** when the agent's visible reply is made benign (as an attacker would), reading the
  reply is blind (AUC **0.454**) while reading the activation still works (**0.993**). The thesis in
  one number.
- **False-positive stress (the key limit):** on 20 legitimate-but-sensitive requests (rotate keys,
  revoke access, download a backup), the probe flags **10 of 20**. It reads *danger*, not
  *permission*, so it is a flag-for-review signal, not an autonomous block.
- **By intent:** exfiltration 98%, evasion 100%, privilege 95%, fraud 88%.

## What's here

| Path | What it is |
|---|---|
| `the-tell.pdf` / `the-tell.tex` | the paper |
| `scripts/` | the full pipeline (dataset → activations → metrics → figures) |
| `models/pack_qwen2.5-1.5b.json` | the intent probe: `{layer, direction, threshold}`, 16 KB |
| `data/results.json` | every number in the paper |
| `figs/` | the paper's figures |

## Reproduce it

Runs on one machine, no GPU. A recent Apple-silicon laptop does the whole thing in a few minutes.

```bash
python3 -m venv .venv
./.venv/bin/pip install torch transformers accelerate safetensors numpy scikit-learn matplotlib huggingface_hub
# models download on first run
./.venv/bin/python scripts/10_make_dataset.py     # 224 requests: benign / explicit / obfuscated
./.venv/bin/python scripts/11_extract.py          # last-token activations, all layers
./.venv/bin/python scripts/12_metrics.py          # per-layer AUC, obfuscated split, text baseline
./.venv/bin/python scripts/13_interpret.py        # what the direction means, in the model's words
./.venv/bin/python scripts/21_second_perlayer.py  # the second model, Qwen2.5-3B
./.venv/bin/python scripts/30_third_model.py      # a third model, different family (Mistral-7B)
./.venv/bin/python scripts/31_latency.py          # the read's cost vs a forward pass
./.venv/bin/python scripts/32_per_category.py     # detection by kind of intent
./.venv/bin/python scripts/33_fp_stress.py        # false positives on legitimate-sensitive requests
./.venv/bin/python scripts/34_deception.py        # reply-blind vs activation-read
./.venv/bin/python scripts/plots.py               # core figures
./.venv/bin/python scripts/plots2.py              # geometry, per-category, fp, latency, three-model
tectonic the-tell.tex                             # build the paper
```

`scripts/01`–`03` are exploratory checks and are not part of the reported pipeline.

## The probe, in code

An intent probe is a direction plus a layer plus a threshold. Reading a request is one dot product:

```python
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

pack = json.load(open("models/pack_qwen2.5-1.5b.json"))
d = np.array(pack["direction"]); L = pack["layer"]; thr = pack["threshold"]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").eval()

def intent_score(system, request):
    enc = tok.apply_chat_template(
        [{"role":"system","content":system},{"role":"user","content":request}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        h = model(enc["input_ids"], output_hidden_states=True).hidden_states[L][0,-1]
    h = h.float().numpy(); h = h/ (np.linalg.norm(h)+1e-6)
    score = float(h @ d)
    return score, ("HOLD" if score > thr else "allow")
```

## What it does not do

Stated plainly in the paper. The read needs the model's weights, so it does not apply to an agent on
a hosted API. The direction is specific to one model. It gives one score per request and does not
point at the guilty span. And against wording that hides intent in ordinary phrasing, it draws level
with a plain text classifier rather than pulling ahead.

## Credit

The read builds on interpretability work on a model's *global workspace*, notably Anthropic's 2026
study, which showed that a model holds a small set of representations it is poised to state and that
they can be read through the model's own unembedding. This work applies that idea to one narrow,
practical question: reading an agent's intent before it acts, on a model you run yourself.
