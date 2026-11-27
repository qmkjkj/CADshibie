#!/usr/bin/env python3
"""Paired frozen-branch ablation of legacy geometry reassignment after native ink."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cadvec.schema import write_output
from cadvec.evaluate import evaluate_directories
from tools.replay_hybrid import replay


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    a=p.parse_args();modes=('no_anchor','no_atoms','no_reassignment')
    paths=sorted((a.base/'stages').glob('*.json'))
    for i,path in enumerate(paths):
        s=json.loads(path.read_text())
        for mode in modes:
            updated=copy.deepcopy(s)
            if mode in ('no_anchor','no_reassignment'):updated['assignment']['anchor_classes']=[]
            if mode in ('no_atoms','no_reassignment'):updated['assignment']['atom_classes']=[]
            write_output(replay(updated),a.base/mode/path.name)
        if (i+1)%25==0:print('assignment',i+1,'/',len(paths),flush=True)
    for mode in modes:
        report=evaluate_directories(a.base/'gt',a.base/mode)
        (a.base/(mode+'_eval.json')).write_text(json.dumps(report,indent=1))
        print(mode,report['micro'],report['TP'],report['FP'],report['FN'],flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
