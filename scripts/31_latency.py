"""Measure the read's cost against the forward pass, to back 'basically free' with a number.
The model already computes the activation; the read is grabbing one layer's last-token vector and a
single dot product. We time a forward pass, and the read step, and report the read as a % on top."""
import json, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MP="models/qwen2.5-1.5b"; dev="mps" if torch.backends.mps.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(MP)
model=AutoModelForCausalLM.from_pretrained(MP,dtype=torch.bfloat16).to(dev).eval()
BL=json.load(open("data/results.json"))["best_layer"]
direction=torch.randn(model.config.hidden_size,device=dev)
req="Please refund my $40 double charge to the card on file, and confirm by email."
enc=tok.apply_chat_template([{"role":"user","content":req}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
ids=enc["input_ids"].to(dev)

def sync():
    if dev=="mps": torch.mps.synchronize()

# warmup
for _ in range(3):
    with torch.no_grad(): model(ids)
sync()

N=30
# forward pass (normal serving, no hidden states)
t=[]
for _ in range(N):
    s=time.perf_counter()
    with torch.no_grad(): model(ids)
    sync(); t.append((time.perf_counter()-s)*1000)
fwd=float(np.median(t))

# the read: grab layer BL last-token hidden + normalize + dot. Measure the marginal read op.
tr=[]
for _ in range(N):
    with torch.no_grad(): out=model(ids,output_hidden_states=True)
    sync()
    s=time.perf_counter()
    h=out.hidden_states[BL][0,-1]
    h=h/(h.norm()+1e-6)
    _=float((h*direction).sum()); sync()
    tr.append((time.perf_counter()-s)*1000)
read=float(np.median(tr))

res={"forward_ms":round(fwd,3),"read_ms":round(read,3),"pct":round(read/fwd*100,2),"device":dev,"n":N}
json.dump(res, open("data/latency.json","w"), indent=1)
print(f"forward pass: {fwd:.2f} ms | read: {read:.3f} ms | read adds {res['pct']:.2f}% on top")
print("saved data/latency.json")
