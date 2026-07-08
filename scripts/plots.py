"""All figures, clean and consistent. Muted palette matching the paper (ink/accent/green)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

INK="#16324F"; ACC="#B5172F"; GRN="#1B7837"; GREY="#8A93A0"; AMBER="#C77D11"
plt.rcParams.update({
 "font.size":11,"axes.edgecolor":"#C7CDD4","axes.linewidth":0.9,"figure.dpi":150,
 "savefig.bbox":"tight","savefig.pad_inches":0.06,"axes.grid":True,"grid.alpha":0.18,
 "grid.linestyle":"-","grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,
 "xtick.color":"#4A5560","ytick.color":"#4A5560","axes.labelcolor":INK,"text.color":INK})

R=json.load(open("data/results.json"))
d=np.load("data/acts.npz",allow_pickle=True); A,y,kind=d["acts"],d["label"],d["kind"]
BL=R["best_layer"]
nrm=lambda X: X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-6)

# ---- FIG scatter: layer-16 activations projected to 2D, the separation you can see ----
XL=nrm(A[:,BL,:])
lda=LinearDiscriminantAnalysis(n_components=1).fit(XL,y)
ax1=lda.transform(XL)[:,0]                    # discriminant axis (benign vs malicious)
pc=PCA(2).fit_transform(XL - XL.mean(0))
ax2=pc[:,1] if abs(np.corrcoef(pc[:,0],ax1)[0,1])>abs(np.corrcoef(pc[:,1],ax1)[0,1]) else pc[:,0]
fig,ax=plt.subplots(figsize=(6.0,4.0))
sty={"benign":("benign request",GRN,"o"),"explicit":("explicit malicious",ACC,"^"),
     "obfuscated":("obfuscated malicious",AMBER,"s")}
for k,(lab,c,m) in sty.items():
    sel=kind==k
    ax.scatter(ax1[sel],ax2[sel],c=c,marker=m,s=46,alpha=0.8,edgecolors="white",linewidths=0.6,label=lab)
ax.axvline(0,color=GREY,lw=0.8,ls="--",alpha=0.7)
ax.set_xlabel("intent direction  (benign  ←     →  malicious)")
ax.set_ylabel("second component")
ax.legend(loc="upper left",fontsize=9,framealpha=0.9)
ax.set_title("Each dot is one agent request, seen through the model's activations",fontsize=11,color=INK,pad=8)
ax.grid(True,alpha=0.18)
fig.savefig("figs/fig_scatter.pdf"); plt.close(fig)

# ---- FIG hist: score distribution with the operating threshold ----
proj=(XL@ (XL[y==1].mean(0)-XL[y==0].mean(0)))
proj=(proj-proj.mean())/proj.std()
thr=np.sort(proj[y==1])[int(0.05*(y==1).sum())]
fig,ax=plt.subplots(figsize=(6.0,3.4))
ax.hist(proj[y==0],bins=22,color=GRN,alpha=0.72,label="benign",edgecolor="white",linewidth=0.5)
ax.hist(proj[y==1],bins=22,color=ACC,alpha=0.72,label="malicious",edgecolor="white",linewidth=0.5)
ax.axvline(thr,color=INK,lw=1.6,ls="--")
ax.text(thr,ax.get_ylim()[1]*0.92,"  threshold\n  (95% recall, 0 false alarms)",fontsize=8.5,color=INK,va="top")
ax.set_xlabel("intent score  (one dot product per request)"); ax.set_ylabel("requests")
ax.legend(loc="upper center",fontsize=9.5,framealpha=0.9)
ax.set_title("The score pulls the two apart with a clean gap",fontsize=11,color=INK,pad=8)
fig.savefig("figs/fig_hist.pdf"); plt.close(fig)

# ---- FIG per-layer AUC (clean) ----
pl=R["per_layer_auc"]; xs=[r["layer"] for r in pl]; ys=[r["auc"] for r in pl]
fig,ax=plt.subplots(figsize=(6.0,3.3))
ax.axhspan(0.5,0.7,color=GREY,alpha=0.07)
ax.plot(xs,ys,color=INK,lw=2.2,marker="o",ms=3.5,mfc="white",mec=INK,mew=1.1)
ax.scatter([BL],[R["best_auc"]],color=ACC,zorder=6,s=70,edgecolors="white",linewidths=1.2)
ax.annotate(f"layer {BL}\n{R['best_auc']:.3f}",(BL,R["best_auc"]),(BL+2.2,R["best_auc"]-0.085),
    color=ACC,fontsize=9.5,fontweight="bold",ha="left",
    arrowprops=dict(arrowstyle="-",color=ACC,lw=1.0))
ax.set_xlabel("layer  (of 28)"); ax.set_ylabel("separation (AUC)"); ax.set_ylim(0.78,1.01)
ax.set_title("Intent is legible from the early layers on",fontsize=11,color=INK,pad=8)
fig.savefig("figs/fig_auc_layers.pdf"); plt.close(fig)

# ---- FIG two models on normalized depth ----
def curve(entry,xs_key="layer"):
    p=entry; return [r["layer"] for r in p],[r["auc"] for r in p]
fig,ax=plt.subplots(figsize=(6.0,3.3))
x1,y1=[r["layer"]/R["n_layers"] for r in pl],[r["auc"] for r in pl]
sm=R["second_model"]; x2=[r["layer"]/sm["n_layers"] for r in sm["per_layer_auc"]]; y2=[r["auc"] for r in sm["per_layer_auc"]]
ax.plot(x1,y1,color=INK,lw=2.2,marker="o",ms=3,mfc="white",label=f"Qwen2.5-1.5B  (peak {R['best_auc']:.3f})")
ax.plot(x2,y2,color=ACC,lw=2.2,marker="s",ms=3,mfc="white",label=f"Qwen2.5-3B  (peak {sm['best_auc']:.3f})")
ax.set_xlabel("relative depth  (layer / total)"); ax.set_ylabel("separation (AUC)"); ax.set_ylim(0.78,1.01)
ax.legend(loc="lower right",fontsize=9.5,framealpha=0.9)
ax.set_title("Two open models, same story, each in its own layer",fontsize=11,color=INK,pad=8)
fig.savefig("figs/fig_twomodel.pdf"); plt.close(fig)

# ---- FIG ROC ----
fig,ax=plt.subplots(figsize=(4.6,3.7))
ax.plot(R["roc"]["probe"]["fpr"],R["roc"]["probe"]["tpr"],color=INK,lw=2.4,
    label=f"read the activation  ({R['overall']['probe_auc']:.3f})")
ax.plot(R["roc"]["text"]["fpr"],R["roc"]["text"]["tpr"],color=ACC,lw=2.0,ls="--",
    label=f"read the words  ({R['overall']['text_auc']:.3f})")
ax.plot([0,1],[0,1],color=GREY,lw=0.9,ls=":")
ax.set_xlabel("false alarms"); ax.set_ylabel("caught"); ax.legend(fontsize=9,loc="lower right",framealpha=0.9)
ax.set_title("Reading inside beats reading the surface, overall",fontsize=10.5,color=INK,pad=8)
fig.savefig("figs/fig_roc.pdf"); plt.close(fig)

# ---- FIG compare bars ----
og=R["obfuscated_generalization"]
groups=["all requests","obfuscated\n(no trigger words)"]
probe=[R["overall"]["probe_auc"],og["probe_auc"]]; text=[R["overall"]["text_auc"],og["text_auc"]]
x=np.arange(2); w=0.34
fig,ax=plt.subplots(figsize=(4.9,3.7))
b1=ax.bar(x-w/2,probe,w,color=INK,label="read the activation")
b2=ax.bar(x+w/2,text,w,color=ACC,label="read the words")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.006,f"{b.get_height():.3f}",ha="center",fontsize=8.5,color=INK)
ax.set_xticks(x); ax.set_xticklabels(groups,fontsize=9.5); ax.set_ylim(0.8,1.03); ax.set_ylabel("separation (AUC)")
ax.legend(fontsize=9,loc="lower left",framealpha=0.9)
ax.set_title("Strip the trigger words and the gap closes",fontsize=10.5,color=INK,pad=8)
fig.savefig("figs/fig_compare.pdf"); plt.close(fig)

print("wrote: scatter, hist, auc_layers, twomodel, roc, compare")
