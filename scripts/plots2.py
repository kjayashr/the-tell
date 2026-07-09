"""Explanatory + new-experiment figures. Geometry of the read (from real data), plus per-category,
transfer, and latency once those data files exist."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

INK="#16324F"; ACC="#B5172F"; GRN="#1B7837"; GREY="#8A93A0"; AMBER="#C77D11"
plt.rcParams.update({"font.size":11,"axes.edgecolor":"#C7CDD4","axes.linewidth":0.9,"figure.dpi":150,
 "savefig.bbox":"tight","savefig.pad_inches":0.06,"axes.grid":False,
 "axes.spines.top":False,"axes.spines.right":False,"xtick.color":"#4A5560","ytick.color":"#4A5560",
 "axes.labelcolor":INK,"text.color":INK})
R=json.load(open("data/results.json"))
d=np.load("data/acts.npz",allow_pickle=True); A,y,kind=d["acts"],d["label"],d["kind"]; BL=R["best_layer"]
nrm=lambda X: X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-6)
XL=nrm(A[:,BL,:])
lda=LinearDiscriminantAnalysis(n_components=1).fit(XL,y)
ax1=lda.transform(XL)[:,0]
if ax1[y==1].mean()<ax1[y==0].mean(): ax1=-ax1   # malicious to the right
pc=PCA(2).fit_transform(XL-XL.mean(0))
ax2=pc[:,0] if abs(np.corrcoef(pc[:,0],ax1)[0,1])<abs(np.corrcoef(pc[:,1],ax1)[0,1]) else pc[:,1]
thr=np.sort(ax1[y==1])[int(0.05*(y==1).sum())]

# ---- FIG geometry: the whole method in one picture ----
fig,ax=plt.subplots(figsize=(6.6,4.3))
for k,(lab,c,m) in {"benign":("safe requests",GRN,"o"),"explicit":("unsafe requests",ACC,"^"),
                    "obfuscated":("unsafe (disguised)",AMBER,"s")}.items():
    s=kind==k; ax.scatter(ax1[s],ax2[s],c=c,marker=m,s=44,alpha=0.78,edgecolors="white",linewidths=0.6,label=lab)
# the separating line (threshold) and the direction arrow
ax.axvline(thr,color=INK,lw=1.4,ls="--")
ax.text(thr,ax2.max()*0.98,"  the line\n  (alarm here)",fontsize=8.5,color=INK,va="top")
y0=ax2.min()-0.05
ax.set_ylim(y0-0.045, ax2.max()+0.02)
ax.annotate("", xy=(ax1.max()*0.98,y0), xytext=(ax1.min()*0.98,y0),
            arrowprops=dict(arrowstyle="-|>",color=INK,lw=2.2))
ax.text(ax1.min()*0.95,y0+0.02,"the probe: the direction from safe to unsafe",fontsize=9.5,color=INK,ha="left",fontweight="bold",va="bottom")
# a new request, projected onto the probe
nx,ny=ax1[y==1][3], ax2[y==1][3]
ax.scatter([nx],[ny],facecolors="none",edgecolors=INK,s=220,linewidths=1.8,zorder=6)
ax.annotate("a new request", (nx,ny),(nx-0.42*(ax1.max()-ax1.min()),ny+0.09),fontsize=9,color=INK,
            arrowprops=dict(arrowstyle="-",color=INK,lw=0.8))
ax.plot([nx,nx],[ny,y0],color=INK,lw=0.9,ls=":")
ax.scatter([nx],[y0],color=INK,s=30,zorder=7)
ax.annotate("its score", (nx,y0),(nx+0.13*(ax1.max()-ax1.min()),y0-0.035),fontsize=8.5,color=INK,
            arrowprops=dict(arrowstyle="-",color=INK,lw=0.6))
ax.legend(loc="upper left",fontsize=9,framealpha=0.9)
ax.set_yticks([]); ax.set_xlabel(""); ax.set_xticks([])
ax.set_title("The whole method in one picture",fontsize=11.5,color=INK,pad=8)
fig.savefig("figs/fig_geometry.pdf"); plt.close(fig)
print("wrote figs/fig_geometry.pdf")

# ---- per-category (if present) ----
if os.path.exists("data/per_category.json"):
    pc_=json.load(open("data/per_category.json"))
    cats=list(pc_.keys()); vals=[pc_[c] for c in cats]
    fig,ax=plt.subplots(figsize=(5.6,3.2))
    ax.barh(cats,vals,color=INK); ax.set_xlim(0,1.03)
    for i,v in enumerate(vals): ax.text(v+0.01,i,f"{v:.0%}",va="center",fontsize=9)
    ax.set_xlabel("caught (recall at the shared alarm line)")
    ax.set_title("What it catches, by kind of intent",fontsize=11,color=INK,pad=6)
    ax.grid(axis="x",alpha=0.2)
    fig.savefig("figs/fig_category.pdf"); plt.close(fig); print("wrote figs/fig_category.pdf")

# ---- FP stress: legitimate-but-sensitive requests scored ----
if os.path.exists("data/fp_stress.json"):
    fp=json.load(open("data/fp_stress.json")); sc=fp["scores"]; th=fp["threshold"]
    order=np.argsort(sc); sc=np.array(sc)[order]
    fig,ax=plt.subplots(figsize=(6.4,3.0))
    cols=[ACC if s>th else GRN for s in sc]
    ax.scatter(sc,np.arange(len(sc)),c=cols,s=60,alpha=0.85,edgecolors="white",linewidths=0.6,zorder=3)
    ax.axvline(th,color=INK,lw=1.4,ls="--")
    ax.text(th,len(sc)-0.5,f" alarm line",fontsize=9,color=INK,va="top")
    ax.text(sc.min(),len(sc)+0.5,"allowed",color=GRN,fontsize=9.5,fontweight="bold")
    ax.text(sc.max(),len(sc)+0.5,"flagged",color=ACC,fontsize=9.5,fontweight="bold",ha="right")
    ax.set_yticks([]); ax.set_xlabel("intent score"); ax.set_ylim(-1,len(sc)+2)
    ax.set_title(f"{fp['false_positives']} of {fp['n']} legitimate, authorized requests were flagged",fontsize=10.5,color=INK,pad=6)
    ax.grid(axis="x",alpha=0.18)
    fig.savefig("figs/fig_fp.pdf"); plt.close(fig); print("wrote figs/fig_fp.pdf")

# ---- transfer heatmap (if present) ----
if os.path.exists("data/transfer.json"):
    t=json.load(open("data/transfer.json")); labels=t["labels"]; M=np.array(t["matrix"])
    fig,ax=plt.subplots(figsize=(4.6,4.0))
    im=ax.imshow(M,cmap="RdBu_r",vmin=0.4,vmax=1.0)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels,fontsize=8,rotation=20,ha="right"); ax.set_yticklabels(labels,fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=8.5,
                    color="white" if abs(M[i,j]-0.7)>0.22 else INK)
    ax.set_xlabel("tested on"); ax.set_ylabel("probe built on")
    ax.set_title("A probe works on its own model, not others",fontsize=10.5,color=INK,pad=6)
    fig.savefig("figs/fig_transfer.pdf"); plt.close(fig); print("wrote figs/fig_transfer.pdf")

# ---- latency bar (if present) ----
if os.path.exists("data/latency.json"):
    lt=json.load(open("data/latency.json"))
    fig,ax=plt.subplots(figsize=(4.4,3.0))
    b=ax.bar(["forward pass\n(what the model\nalready does)","the read\n(our added cost)"],
             [lt["forward_ms"],lt["read_ms"]],color=[GREY,INK],width=0.55)
    for r_ in b: ax.text(r_.get_x()+r_.get_width()/2,r_.get_height(),f"{r_.get_height():.2f} ms",ha="center",va="bottom",fontsize=9)
    ax.set_ylabel("time per request (ms)")
    ax.set_title(f"The read adds about {lt['pct']:.1f}% on top",fontsize=10.5,color=INK,pad=6)
    ax.grid(axis="y",alpha=0.2)
    fig.savefig("figs/fig_latency.pdf"); plt.close(fig); print("wrote figs/fig_latency.pdf")

# ---- three-model curve (if third model present) ----
if "third_model" in R:
    fig,ax=plt.subplots(figsize=(6.0,3.3)); plt.rcParams["axes.grid"]=True
    def curve(entry,nl,c,mk,lab):
        xs=[r["layer"]/nl for r in entry]; ys=[r["auc"] for r in entry]
        ax.plot(xs,ys,color=c,lw=2.0,marker=mk,ms=3,mfc="white",label=lab)
    curve(R["per_layer_auc"],R["n_layers"],INK,"o",f"Qwen2.5-1.5B ({R['best_auc']:.3f})")
    curve(R["second_model"]["per_layer_auc"],R["second_model"]["n_layers"],ACC,"s",f"Qwen2.5-3B ({R['second_model']['best_auc']:.3f})")
    tm=R["third_model"]; curve(tm["per_layer_auc"],tm["n_layers"],GRN,"^",f"{tm['name']} ({tm['best_auc']:.3f})")
    ax.set_xlabel("relative depth (layer / total)"); ax.set_ylabel("separation (AUC)"); ax.set_ylim(0.75,1.01)
    ax.legend(loc="lower right",fontsize=8.5,framealpha=0.9); ax.grid(alpha=0.18)
    ax.set_title("Three open models, two families, same result",fontsize=11,color=INK,pad=6)
    fig.savefig("figs/fig_threemodel.pdf"); plt.close(fig); print("wrote figs/fig_threemodel.pdf")
