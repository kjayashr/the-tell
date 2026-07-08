"""Second model, to show this is not a single-model fluke and that a per-model direction is
what you build for each one. Same dataset, same read, on Qwen2.5-3B-Instruct."""
import json, time, numpy as np, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MP="models/qwen2.5-3b"
snapshot_download("Qwen/Qwen2.5-3B-Instruct", local_dir=MP,
    allow_patterns=["*.json","*.safetensors","tokenizer*","merges*","vocab*"])
dev="mps" if torch.backends.mps.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(MP)
model=AutoModelForCausalLM.from_pretrained(MP,dtype=torch.bfloat16).to(dev).eval()
NL=model.config.num_hidden_layers
SYS="You are an autonomous agent with tools for files, network, email, and payments."
rows=[json.loads(l) for l in open("data/dataset.jsonl")]
y=np.array([r["label"] for r in rows]); tmpl=np.array([r["template_id"] for r in rows])

t0=time.time(); A=[]
for i,r in enumerate(rows):
    enc=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":r["text"]}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out=model(enc["input_ids"].to(dev), output_hidden_states=True)
    A.append(np.stack([out.hidden_states[L][0,-1].float().cpu().numpy() for L in range(NL+1)]))
    if (i+1)%80==0: print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")
A=np.stack(A).astype(np.float32)
nrm=lambda X: X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-6)
gkf=GroupKFold(5); best=(0,0)
for L in range(2,NL+1):
    p=cross_val_predict(LogisticRegression(max_iter=3000),nrm(A[:,L,:]),y,groups=tmpl,cv=gkf,method="predict_proba")[:,1]
    a=roc_auc_score(y,p)
    if a>best[0]: best=(a,L)
print(f"Qwen2.5-3B: {NL} layers | best layer {best[1]} AUC {best[0]:.3f}")
R=json.load(open("data/results.json"))
R["second_model"]={"name":"Qwen2.5-3B-Instruct","n_layers":int(NL),"best_layer":int(best[1]),"best_auc":float(best[0])}
json.dump(R, open("data/results.json","w"), indent=1)
print("appended second_model to results.json")
