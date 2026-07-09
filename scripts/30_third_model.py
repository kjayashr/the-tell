"""Third model, a different family from Qwen. Try Mistral-7B-Instruct-v0.3 (ungated); if that is
gated or too big, fall back to Phi-3.5-mini-instruct. Same 224 requests, same held-out probe,
full per-layer AUC curve. Writes data/acts_third.npz and results.json['third_model']."""
import json, time, numpy as np, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CANDIDATES = [("Mistral-7B-Instruct-v0.3","mistralai/Mistral-7B-Instruct-v0.3","models/mistral-7b"),
              ("Phi-3.5-mini-instruct","microsoft/Phi-3.5-mini-instruct","models/phi-3.5-mini")]
name=repo=local=None
for nm,rp,lo in CANDIDATES:
    try:
        snapshot_download(rp, local_dir=lo, allow_patterns=["*.json","*.safetensors","tokenizer*","*.model","merges*","vocab*"])
        name,repo,local=nm,rp,lo; break
    except Exception as e:
        print(f"skip {nm}: {str(e)[:80]}")
assert local, "no third model available"
print("using", name)

dev="mps" if torch.backends.mps.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(local)
model=AutoModelForCausalLM.from_pretrained(local,dtype=torch.bfloat16).to(dev).eval()
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
np.savez_compressed("data/acts_third.npz", acts=A, label=y, template=tmpl)
nrm=lambda X: X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-6); gkf=GroupKFold(5)
pl=[]
for L in range(NL+1):
    p=cross_val_predict(LogisticRegression(max_iter=3000),nrm(A[:,L,:]),y,groups=tmpl,cv=gkf,method="predict_proba")[:,1]
    pl.append({"layer":int(L),"auc":float(roc_auc_score(y,p))})
best=max(pl,key=lambda r:r["auc"])
R=json.load(open("data/results.json"))
R["third_model"]={"name":name,"n_layers":int(NL),"best_layer":int(best["layer"]),"best_auc":float(best["auc"]),"per_layer_auc":pl}
json.dump(R, open("data/results.json","w"), indent=1)
print(f"{name}: {NL} layers | best layer {best['layer']} AUC {best['auc']:.3f} ({time.time()-t0:.0f}s)")
