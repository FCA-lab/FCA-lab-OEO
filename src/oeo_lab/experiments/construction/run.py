"""Run with python -m oeo_lab.experiments.construction.run --help."""
import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
from .datasets import load_context
from .evaluate import execute
from ..common.checkpoints import digest, save
from ..common.runner import run_tasks

def tasks_for(config, root, suite):
    backgrounds = []
    if suite in ("all", "example"):
        backgrounds.append(copy.deepcopy(config["example"]))
    if suite in ("all", "real"):
        for entry in config["real_datasets"]:
            entry = copy.deepcopy(entry)
            entry["path"] = str((root / entry["path"]).resolve())
            entry["source_sha256"] = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
            backgrounds.append(entry)
    if suite in ("all", "synthetic"):
        grid = config["synthetic"]
        for m in grid["objects"]:
            for n in grid["attributes"]:
                for d in grid["densities"]:
                    hashes = set()
                    for repeat in range(grid["repeats"]):
                        seed = int(digest([grid["base_seed"],m,n,d,repeat])[:16],16)
                        entry = {"name": f"O{m}_A{n}_D{d}_R{repeat}", "kind":"synthetic",
                                 "objects":m,"attributes":n,"density":d,"seed":seed,"repeat":repeat}
                        # Verify distinct realized backgrounds without storing matrices.
                        fingerprint = digest(load_context(entry))
                        if fingerprint in hashes:
                            raise ValueError("Duplicate random background; change base_seed or grid")
                        hashes.add(fingerprint)
                        backgrounds.append(entry)
    return [{"id":digest([b,a])[:24],"background":b,"algorithm":a} for b in backgrounds for a in config["algorithms"]]

def summarize(tasks, directory, output, template):
    import pandas as pd
    rows, paired = [], {}
    for task in tasks:
        path = directory / template.format(task_id=task["id"])
        record = json.loads(path.read_text()) if path.exists() else {"status":"pending"}
        b = task["background"]
        row = {"task_id":task["id"],"background":b["name"],"kind":b["kind"],
               "algorithm":task["algorithm"],"repeat":b.get("repeat",0),"seed":str(b.get("seed","")),
               "target_density":b.get("density"),"status":record["status"],"error":record.get("error","")}
        if record["status"] == "ok":
            result = record["result"]
            row.update({k:v for k,v in result.items() if k != "concepts"})
            paired.setdefault(b["name"], {})[task["algorithm"]] = result
        rows.append(row)
    comparisons = []
    for name in dict.fromkeys(t["background"]["name"] for t in tasks):
        pair = paired.get(name,{})
        if set(pair) != {"3woc","qm"}:
            comparisons.append({"background":name,"status":"incomplete"}); continue
        a,b = pair["3woc"],pair["qm"]
        left,right = set(map(tuple,a["concepts"])),set(map(tuple,b["concepts"]))
        passed = a["background_sha256"] == b["background_sha256"] and left == right and not any(x["invalid_concepts"] or x["duplicate_concepts"] for x in (a,b))
        comparisons.append({"background":name,"status":"pass" if passed else "fail",
                            "only_3woc":len(left-right),"only_qm":len(right-left)})
    frame = pd.DataFrame(rows)
    summaries = []
    groups = {}
    for row,task in zip(rows,tasks):
        b = task["background"]
        key = (b["kind"],row.get("objects",b.get("objects")),row.get("attributes",b.get("attributes")),b.get("density",row.get("actual_density")),
               "synthetic" if b["kind"]=="synthetic" else b["name"],task["algorithm"])
        groups.setdefault(key,[]).append(row)
    for key,group in groups.items():
        valid=[r for r in group if r["status"]=="ok"]
        summary=dict(zip(["kind","objects","attributes","density","dataset","algorithm"],key))
        summary.update(expected_runs=len(group),completed_runs=len(valid),complete=len(valid)==len(group))
        for field in ["seconds","concept_count","result_container_bytes"]:
            values=pd.Series([r[field] for r in valid],dtype=float)
            # Do not silently report a partial group as the requested five-run mean.
            summary[field+"_mean"]=values.mean() if len(valid)==len(group) else None
            summary[field+"_std"]=values.std(ddof=1) if len(valid)==len(group) and len(valid)>1 else None
        summaries.append(summary)
    tmp=output.with_name(output.stem+".tmp.xlsx")
    with pd.ExcelWriter(tmp,engine="openpyxl") as writer:
        frame.to_excel(writer,sheet_name="runs",index=False)
        pd.DataFrame(summaries).to_excel(writer,sheet_name="summary",index=False)
        pd.DataFrame(comparisons).to_excel(writer,sheet_name="comparisons",index=False)
    os.replace(tmp,output)
    return comparisons

def main():
    root=Path(__file__).resolve().parents[4]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--suite",choices=["all","example","real","synthetic"],default="all")
    parser.add_argument("--n-jobs",type=int)
    parser.add_argument("--timeout",type=float)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    if args.n_jobs is not None: config["execution"]["n_jobs"]=args.n_jobs
    if args.timeout is not None: config["execution"]["task_timeout_sec"]=args.timeout
    if config["execution"]["n_jobs"]<1: parser.error("n-jobs must be positive")
    if set(config["algorithms"])!={"3woc","qm"} or len(config["algorithms"])!=2:
        parser.error("Reproduction requires exactly 3woc and qm")
    tasks=tasks_for(config,root,args.suite)
    print(f"{len(tasks)} algorithm tasks; {len(tasks)//2} backgrounds",flush=True)
    if args.dry_run:return
    for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[var]="1"
    # Only computation code determines reuse; reporting-only edits do not retrain.
    source_files=list((root/"src/fca_core").glob("*.py"))
    source_files+=list((root/"src/oeo_lab/algorithms/construction").glob("*.py"))
    source_files += [Path(__file__).with_name(n) for n in ("datasets.py", "evaluate.py")]
    source_hash=digest({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    provenance={"code_sha256":source_hash,"python":platform.python_version(),
                "platform":platform.platform(),"n_jobs":config["execution"]["n_jobs"],
                "memory_metric":"final_container_shallow_v1"}
    signature=digest(provenance)
    for task in tasks:
        task["signature"]=signature
        task["id"]=digest([task["background"],task["algorithm"],signature])[:24]
    out=root/config["output"]["root"]
    out.mkdir(parents=True,exist_ok=True)
    files=config["output"]["files"]
    directory=out/config["output"]["checkpoint_dir"]
    index_path=directory/config["output"]["checkpoint_index"]
    # Index is recovery metadata only, never an experiment configuration source.
    snapshot=json.loads(index_path.read_text()) if index_path.exists() else {"tasks":[],"provenance":{}}
    # One active result per logical task. Older checkpoints remain recoverable,
    # but incompatible computation versions are not mixed into the workbook.
    registry={ (t["background"]["name"],t["algorithm"]):t for t in snapshot["tasks"] if t.get("signature")==signature }
    registry.update({(t["background"]["name"],t["algorithm"]):t for t in tasks})
    all_tasks=list(registry.values())
    snapshot.update(tasks=all_tasks)
    snapshot["provenance"][signature]=provenance
    save(index_path,snapshot)
    template=config["output"]["checkpoint_template"]
    try:
        run_tasks(tasks,execute,directory,config["execution"],template)
    finally:
        comparisons = []
        suite_kinds = {"real": "categorical", "synthetic": "synthetic", "example": "matrix"}
        for suite, kind in suite_kinds.items():
            suite_tasks = [t for t in all_tasks if t["background"]["kind"] == kind]
            if suite_tasks:
                comparisons.extend(summarize(suite_tasks, directory,
                                             out/files["results"][suite], template))
    print(f"Results: {out}")
    if any(row["status"]!="pass" for row in comparisons):
        raise SystemExit("Some comparisons failed or remain incomplete; see results and checkpoints")

if __name__=="__main__":
    main()
