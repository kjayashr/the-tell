"""Interesting data-driven figures: high-dim->1D collapse, per-category radar, latency timeline,
and the vocabulary the direction lights up. Light palette, consistent with the rest."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

INK="#16324F"; ACC="#B5172F"; GRN="#1B7837"; GREY="#8A93A0"; AMB="#C77D11"
plt.rcParams.update({"font.size":11,"axes.edgecolor":"#C7CDD4","axes.linewidth":0.9,"figure.dpi":150,
 "savefig.bbox":"tight","savefig.pad_inches":0.08,"text.color":INK,"axes.labelcolor":INK,
 "xtick.color":"#4A5560","ytick.color":"#4A5560","axes.spines.top":False,"axes.spines.right":False})
R=json.load(open("data/results.json")); BL=R["best_layer"]
d=np.load("data/acts.npz",allow_pickle=True); A,y,kind=d["acts"],d["label"],d["kind"]
nrm=lambda X: X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-6); XL=nrm(A[:,BL,:])
lda=LinearDiscriminantAnalysis(n_components=1).fit(XL,y); s1=lda.transform(XL)[:,0]
if s1[y==1].mean()<s1[y==0].mean(): s1=-s1
s1=(s1-s1.mean())/s1.std()
pc=PCA(2).fit_transform(XL-XL.mean(0)); s2=pc[:,0] if abs(np.corrcoef(pc[:,0],s1)[0,1])<abs(np.corrcoef(pc[:,1],s1)[0,1]) else pc[:,1]
s2=(s2-s2.mean())/s2.std(); thr=np.sort(s1[y==1])[int(0.05*(y==1).sum())]
STY={"benign":("safe",GRN,"o"),"explicit":("unsafe",ACC,"^"),"obfuscated":("unsafe (disguised)",AMB,"s")}

# ---- COLLAPSE: 1536 numbers -> 1 ----
from matplotlib.gridspec import GridSpec
fig=plt.figure(figsize=(6.6,4.9)); gs=GridSpec(2,1,height_ratios=[3.0,1.0],hspace=0.55)
axT=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1])
for k,(lab,c,m) in STY.items():
    sel=kind==k
    axT.scatter(s1[sel],s2[sel],c=c,marker=m,s=42,alpha=0.8,edgecolors="white",linewidths=0.5,label=lab)
axT.set_xticks([]); axT.set_yticks([]); axT.set_ylabel("")
for sp in axT.spines.values(): sp.set_visible(True); sp.set_color("#C7CDD4")
axT.legend(loc="lower center",ncol=3,fontsize=8.5,framealpha=0.95,bbox_to_anchor=(0.5,-0.02))
axT.set_title("each request is a list of ~1,500 numbers: one point in this space",fontsize=10.5,color=INK,pad=8)
# bottom: the collapsed number line
for k,(lab,c,m) in STY.items():
    sel=kind==k
    axB.scatter(s1[sel],np.zeros(sel.sum()),c=c,marker=m,s=34,alpha=0.85,edgecolors="white",linewidths=0.4,clip_on=False)
axB.axvline(thr,color=INK,ls="--",lw=1.3); axB.text(thr,0.6,"alarm line",fontsize=8.5,ha="center",color=INK)
axB.set_yticks([]); axB.set_ylim(-1,1); axB.set_xlim(s1.min()-0.3,s1.max()+0.3)
for sp in ["left","right","top"]: axB.spines[sp].set_visible(False)
axB.set_xlabel("one score per request");
axB.text(s1.min(),-0.75,"safe",color=GRN,fontsize=10,fontweight="bold")
axB.text(s1.max(),-0.75,"unsafe",color=ACC,fontsize=10,fontweight="bold",ha="right")
axB.set_title("the probe collapses that whole list to ONE number",fontsize=10.5,color=INK,pad=4)
fig.savefig("figs/fig_collapse.pdf"); plt.close(fig); print("wrote fig_collapse.pdf")

# ---- RADAR per category ----
pcj=json.load(open("data/per_category.json")); cats=list(pcj.keys()); vals=[pcj[c] for c in cats]
N=len(cats); ang=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); ang+=ang[:1]; vv=vals+vals[:1]
fig=plt.figure(figsize=(5.0,4.6)); ax=plt.subplot(111,polar=True)
ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats,fontsize=11,color=INK)
ax.set_ylim(0.7,1.0); ax.set_yticks([0.8,0.9,1.0]); ax.set_yticklabels(["80%","90%","100%"],fontsize=8,color=GREY)
ax.plot(ang,vv,color=INK,lw=2.2); ax.fill(ang,vv,color=INK,alpha=0.15)
for a_,v_,c_ in zip(ang[:-1],vals,cats):
    ax.scatter(a_,v_,color=ACC if v_<0.9 else INK,s=45,zorder=5)
    ax.text(a_,v_+0.015,f"{v_:.0%}",ha="center",fontsize=9,color=ACC if v_<0.9 else INK,fontweight="bold")
ax.set_title("What it catches, by intent",fontsize=12,color=INK,pad=18)
fig.savefig("figs/fig_radar.pdf"); plt.close(fig); print("wrote fig_radar.pdf")

# ---- LATENCY timeline: before the first word ----
lt=json.load(open("data/latency.json")); fwd=lt["forward_ms"]; rd=lt["read_ms"]
fig,ax=plt.subplots(figsize=(6.6,2.4))
ax.barh([0],[fwd],color="#DCE3EA",edgecolor=INK,height=0.5)
ax.barh([0],[rd],left=fwd*0.62,color=AMB,edgecolor=INK,height=0.5)   # placed inside the pass
ax.annotate(f"the read: {rd:.1f} ms",(fwd*0.62+rd/2,0),(fwd*0.62,0.55),fontsize=9,color=AMB,fontweight="bold",
            ha="center",arrowprops=dict(arrowstyle="-",color=AMB,lw=1))
ax.text(fwd/2,0,f"one forward pass: {fwd:.0f} ms",ha="center",va="center",fontsize=10,color=INK)
ax.scatter([fwd],[0],marker="|",s=400,color=GRN)
ax.annotate("first word\nnot yet written",(fwd,0),(fwd+2,-0.5),fontsize=9,color=GRN,ha="left",
            arrowprops=dict(arrowstyle="-",color=GRN,lw=1))
ax.set_xlim(-1,fwd+14); ax.set_ylim(-0.9,0.9); ax.set_yticks([]); ax.set_xlabel("milliseconds")
ax.set_title(f"The read happens inside the forward pass, before the reply , +{lt['pct']:.1f}%",fontsize=11,color=INK,pad=8)
ax.spines["left"].set_visible(False)
fig.savefig("figs/fig_latency_timeline.pdf"); plt.close(fig); print("wrote fig_latency_timeline.pdf")

# ---- VOCAB the direction lights up: clean ranked bars ----
import matplotlib
words=[("forbidden",1.0),("illegal / crime",0.93),("evade / circumvent",0.86),
       ("strictly forbidden",0.78),("insidious",0.70),("traps",0.6)]
fig,ax=plt.subplots(figsize=(6.4,3.0))
yv=np.arange(len(words))[::-1]
for (w,wt),yy in zip(words,yv):
    ax.barh(yy,wt,height=0.62,color=matplotlib.colors.to_rgba(ACC,0.22+0.5*wt),edgecolor=ACC,linewidth=1.2)
    ax.text(0.02,yy,w,va="center",ha="left",fontsize=13,color=INK,
            fontweight="bold" if wt>0.85 else "normal")
ax.set_xlim(0,1.05); ax.set_ylim(-0.6,len(words)-0.4); ax.set_yticks([]); ax.set_xticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("the words the intent direction points at",fontsize=12.5,color=INK,pad=8)
ax.text(0.0,-0.9,"read straight out of the model's own vocabulary",fontsize=9,color=GREY,style="italic")
fig.savefig("figs/fig_vocab.pdf"); plt.close(fig); print("wrote fig_vocab.pdf")
