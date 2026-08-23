"""Confirm the clipper and the duals start from the IDENTICAL cached checkpoint,
so the only difference between the arms is the extra optimiser epochs."""
import glob, json, os
import pandas as pd
rows=[]
for p in glob.glob("results/pending_runs/paper_final/**/config.json", recursive=True):
    c=json.load(open(p)); hp=c.get("hyperparams") or {}; r=c.get("results") or {}
    rows.append({"model":c.get("model_name"),"dataset":c.get("dataset_mode"),
        "cap":c.get("constraint_tag"),"seed":hp.get("seed"),
        "method":c.get("methodology"),"base_model_id":c.get("base_model_id"),
        "used_cached":r.get("used_cached_model"),
        "warmup_time":r.get("warmup_time"),"ctrain_time":r.get("constraint_train_time"),
        "samples_adjusted":r.get("samples_adjusted")})
d=pd.DataFrame(rows)
print("rows",len(d))
g=d.groupby(["model","dataset","seed"]).base_model_id.nunique()
print("\n(model,dataset,seed) groups: %d   groups with >1 distinct base_model_id: %d"
      % (len(g), int((g>1).sum())))
print("=> every method in a (model,dataset,seed) group starts from the SAME cached warm-up checkpoint"
      if (g>1).sum()==0 else "=> checkpoints DIFFER")
print("\nwall-clock ledger, median seconds:")
print(d.groupby("method")[["warmup_time","ctrain_time"]].median().round(2).to_string())
print("\nused_cached_model:")
print(d.groupby("method").used_cached.value_counts().to_string())
