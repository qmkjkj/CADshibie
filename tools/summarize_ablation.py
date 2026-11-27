#!/usr/bin/env python3
"""Summarize paired page evaluations on historical and source-isolated roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.evaluate import _metrics


def aggregate(rows):
    values = np.sum([[r[k] for k in ("TP","FP","FN","sum_iou")] for r in rows],axis=0)
    tp,fp,fn,iou = values if len(rows) else (0,0,0,0)
    return {"images":len(rows),"TP":int(tp),"FP":int(fp),"FN":int(fn),
            "sum_iou":float(iou),**_metrics(int(tp),int(fp),int(fn),float(iou))}


def paired_group_bootstrap(reports, stems, groups, draws=2000, seed=73):
    """Resample entire source components and recompute pooled micro PQ."""
    if not stems or draws <= 0:
        return {}
    missing = set(stems)-set(groups)
    if missing:
        raise ValueError(f"Missing source groups for {len(missing)} pages")
    names = list(reports)
    families = sorted({groups[s] for s in stems})
    family_index = {family:i for i,family in enumerate(families)}
    matrices = {}
    for name,report in reports.items():
        matrix = np.zeros((len(families),4))
        for stem in sorted(stems):
            row = report['per_image'][stem]
            matrix[family_index[groups[stem]]] += [row[k] for k in ('TP','FP','FN','sum_iou')]
        matrices[name] = matrix
    samples = np.random.default_rng(seed).multinomial(
        len(families), np.full(len(families),1/len(families)), size=draws)
    scores = {}
    for name,matrix in matrices.items():
        totals = samples @ matrix
        denominator = totals[:,0]+.5*(totals[:,1]+totals[:,2])
        scores[name] = np.divide(totals[:,3],denominator,
                                 out=np.zeros(draws),where=denominator>0)
    return {'method':'paired source-component percentile bootstrap',
            'source_components':len(families),'draws':draws,'seed':seed,
            'delta_PQ_points_95pct_CI':{
                name:np.percentile(100*(scores[name]-scores[names[0]]),[2.5,97.5]).tolist()
                for name in names[1:]},
            'note':'Conditional on fixed weights/configuration and observed candidate source groups; does not remove historical model exposure.'}


def summarize(reports, splits, bootstrap=2000):
    names = list(reports)
    shared = set(reports[names[0]]["per_image"])
    if any(set(report["per_image"]) != shared for report in reports.values()):
        raise ValueError("Paired experiments must evaluate the exact same pages")
    sets = {"all_evaluated":shared,
            "source_isolated_all":shared & set().union(*map(set,splits["test_final"].values()))}
    for role,stems in splits["test_final"].items():
        sets["source_isolated_"+role] = shared & set(stems)
    result = {}; uncertainty = {}
    for scope,stems in sets.items():
        rows = {name:aggregate([report["per_image"][s] for s in sorted(stems)])
                for name,report in reports.items()}
        baseline = rows[names[0]]["PQ"]
        for name in rows:
            rows[name]["delta_PQ_points"] = 100*(rows[name]["PQ"]-baseline)
        result[scope] = rows
        uncertainty[scope] = paired_group_bootstrap(
            reports,stems,splits['groups']['test_final'],draws=bootstrap)
    return {"baseline":names[0],"scopes":result,
            "paired_uncertainty":uncertainty,
            "note":"Historical frozen weights may have prior label exposure; source isolation does not undo that history."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append",required=True,help="NAME=EVALUATION_JSON, baseline first")
    parser.add_argument("--splits",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=2000)
    args = parser.parse_args()
    reports = {}
    for value in args.eval:
        name,separator,path = value.partition("=")
        if not separator or name in reports:
            parser.error("Use unique NAME=EVALUATION_JSON")
        reports[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    output = summarize(reports,json.loads(args.splits.read_text(encoding="utf-8")),args.bootstrap)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=1),encoding="utf-8")
    for scope,rows in output["scopes"].items():
        for name,row in rows.items():
            print(scope,name,row["images"],f"PQ {100*row['PQ']:.3f}%",f"delta {row['delta_PQ_points']:+.3f}",
                  row['TP'],row['FP'],row['FN'])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
