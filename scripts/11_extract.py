"""Extract the last-token activation at every layer for each row. Cache to data/acts.npz.
This is the read: run the prompt through the model, take the final position's activation.
It happens over the prompt, before any reply, so it is a pre-reply read."""
import json, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MP="models/qwen2.5-1.5b"; dev="mps" if torch.backends.mps.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(MP)
model=AutoModelForCausalLM.from_pretrained(MP,dtype=torch.bfloat16).to(dev).eval()
NL=model.config.num_hidden_layers
SYS="You are an autonomous agent with tools for files, network, email, and payments."

rows=[json.loads(l) for l in open("data/dataset.jsonl")]
t0=time.time(); A=[]
for i,r in enumerate(rows):
    enc=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":r["text"]}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids=enc["input_ids"].to(dev)
    with torch.no_grad():
        out=model(ids, output_hidden_states=True)
    A.append(np.stack([out.hidden_states[L][0,-1].float().cpu().numpy() for L in range(NL+1)]))
    if (i+1)%40==0: print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)")
A=np.stack(A).astype(np.float32)  # [N, NL+1, d]
np.savez_compressed("data/acts.npz", acts=A,
    label=np.array([r["label"] for r in rows]),
    kind=np.array([r["kind"] for r in rows]),
    template=np.array([r["template_id"] for r in rows]),
    text=np.array([r["text"] for r in rows], dtype=object))
print(f"saved acts {A.shape} in {time.time()-t0:.0f}s")
