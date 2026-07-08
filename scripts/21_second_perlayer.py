"""Cache Qwen2.5-3B last-token activations and its full per-layer AUC curve, so we can draw a
two-model figure. Writes data/acts_3b.npz and results.json['second_model']['per_layer_auc']."""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MP="models/qwen2.5-3b"; dev="mps" if torch.backends.mps.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(MP)
model=AutoModelForCausalLM.from_pretrained(MP,dtype=torch.bfloat16).to(dev).eval()
NL=model.config.num_hidden_layers
SYS="You are an autonomous agent with tools for files, network, email, and payments."
rows=[json.loads(l) for l in open("data/dataset.jsonl")]
y=np.array([r["label"] for r in rows]); tmpl=np.array([r["template_id"] for r in rows]); kind=np.array([r["kind"] for r in rows])
A=[]
for r in rows:
    enc=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":r["text"]}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out=model(enc["input_ids"].to(dev), output_hidden_states=True)
    A.append(np.stack([out.hidden_states[L][0,-1].float().cpu().numpy() for L in range(NL+1)]))
A=np.stack(A).astype(np.float32)
np.savez_compressed("data/acts_3b.npz", acts=A, label=y, kind=kind, template=tmpl)
nrm=lambda X: X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-6); gkf=GroupKFold(5)
pl=[]
for L in range(NL+1):
    p=cross_val_predict(LogisticRegression(max_iter=3000),nrm(A[:,L,:]),y,groups=tmpl,cv=gkf,method="predict_proba")[:,1]
    pl.append({"layer":int(L),"auc":float(roc_auc_score(y,p))})
R=json.load(open("data/results.json")); R["second_model"]["per_layer_auc"]=pl
json.dump(R, open("data/results.json","w"), indent=1)
print(f"3B per-layer cached, best {max(pl,key=lambda r:r['auc'])}")
