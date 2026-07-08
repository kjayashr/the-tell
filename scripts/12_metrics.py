"""Compute the results. No model load, just the cached activations.
- per-layer AUC, held out by template (GroupKFold): learn intent, not phrasing.
- the hard test: train on explicit+benign, test on OBFUSCATED+benign (unseen wording, no keywords).
- a text baseline (TF-IDF + logistic) on the same splits, to beat on the obfuscated cases.
- operating point: FPR at 95% recall, recall at 1% FPR.
"""
import json, numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve

d=np.load("data/acts.npz", allow_pickle=True)
A,y,kind,tmpl,text = d["acts"],d["label"],d["kind"],d["template"],d["text"]
NLp1=A.shape[1]
nrm=lambda X: X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-6)
gkf=GroupKFold(5)

# 1) per-layer AUC, GroupKFold by template
per_layer=[]
for L in range(NLp1):
    XL=nrm(A[:,L,:])
    p=cross_val_predict(LogisticRegression(max_iter=3000), XL, y, groups=tmpl, cv=gkf, method="predict_proba")[:,1]
    per_layer.append({"layer":int(L),"auc":float(roc_auc_score(y,p))})
best=max(per_layer,key=lambda r:r["auc"]); BL=best["layer"]
print("per-layer AUC (held out by template):")
for r in per_layer[2:]:
    print(f"  L{r['layer']:2d}: {r['auc']:.3f} {'#'*int(r['auc']*40)}")
print(f"BEST layer {BL}  AUC {best['auc']:.3f}\n")

# overall proba at best layer for operating point + text baseline (Split A)
XB=nrm(A[:,BL,:])
pB=cross_val_predict(LogisticRegression(max_iter=3000), XB, y, groups=tmpl, cv=gkf, method="predict_proba")[:,1]
txt_pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1), LogisticRegression(max_iter=3000))
pT=cross_val_predict(txt_pipe, text, y, groups=tmpl, cv=gkf, method="predict_proba")[:,1]
overall={"probe_auc":float(roc_auc_score(y,pB)),"text_auc":float(roc_auc_score(y,pT))}

fpr,tpr,_=roc_curve(y,pB)
fpr_at_95=float(fpr[np.argmax(tpr>=0.95)]); tpr_at_1=float(tpr[np.argmax(fpr>=0.01)])

# 2) the hard test: train on explicit+benign, test on obfuscated+benign (disjoint benign templates)
ben=sorted(set(tmpl[kind=="benign"]))
trB,teB=set(ben[::2]),set(ben[1::2])
tr=(kind=="explicit")|((kind=="benign")&np.isin(tmpl,list(trB)))
te=(kind=="obfuscated")|((kind=="benign")&np.isin(tmpl,list(teB)))
def auc_on(feat_tr,feat_te,est):
    est.fit(feat_tr,y[tr]);
    p=est.predict_proba(feat_te)[:,1]; return float(roc_auc_score(y[te],p))
hard_probe=auc_on(nrm(A[tr,BL,:]),nrm(A[te,BL,:]),LogisticRegression(max_iter=3000))
hard_text =auc_on(text[tr],text[te],make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1),LogisticRegression(max_iter=3000)))

res={"model":"Qwen2.5-1.5B-Instruct","n":int(len(y)),"n_layers":int(NLp1-1),
     "counts":{k:int((kind==k).sum()) for k in ["benign","explicit","obfuscated"]},
     "per_layer_auc":per_layer,"best_layer":BL,"best_auc":best["auc"],
     "overall":overall,"fpr_at_95tpr":fpr_at_95,"tpr_at_1pct_fpr":tpr_at_1,
     "obfuscated_generalization":{"probe_auc":hard_probe,"text_auc":hard_text,
        "note":"train on explicit+benign, test on obfuscated(no keywords)+benign"},
     "roc":{"probe":{"fpr":fpr.tolist(),"tpr":tpr.tolist()}}}
# text ROC for the plot
ft,tt,_=roc_curve(y,pT); res["roc"]["text"]={"fpr":ft.tolist(),"tpr":tt.tolist()}
json.dump(res, open("data/results.json","w"), indent=1)

print(f"overall @L{BL}:  probe AUC {overall['probe_auc']:.3f}   text-baseline AUC {overall['text_auc']:.3f}")
print(f"operating point: FPR {fpr_at_95*100:.1f}% at 95% recall | recall {tpr_at_1*100:.1f}% at 1% FPR")
print(f"HARD (obfuscated, unseen, no keywords):  probe AUC {hard_probe:.3f}   text-baseline AUC {hard_text:.3f}")
print("saved data/results.json")
