import csv
import json
import sys

from tools.build_expansion_splits import main


def test_expansion_protects_entire_retained_and_already_trained_families(tmp_path,monkeypatch):
    groups={'old':'trained','old_copy':'trained','proxy':'proxy_family',
            'retained':'retained_family','retained_copy':'retained_family',
            'confirm':'confirm_family','confirm_copy':'confirm_family',
            '3UElq':'viewed_family','viewed_copy':'viewed_family',
            'new_a':'new_family_a','new_a_copy':'new_family_a','new_b':'new_family_b'}
    source={'groups':{'train_final':groups,'test_final':{'historical':'new_family_a'}},
            'train_final':{'train':['old'],'val':['proxy','retained']}}
    metadata={'selected_pages':{'train':['old'],'val':['proxy']}}
    protocol={'incremental_confirmation':['confirm']}
    for name,value in (('source',source),('native',metadata),('protocol',protocol)):
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    directory=tmp_path/'data/train_final';directory.mkdir(parents=True)
    with (directory/'manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dest_stem']);writer.writeheader()
        writer.writerows({'dest_stem':n} for n in groups)
    out=tmp_path/'expansion.json'
    monkeypatch.setattr(sys,'argv',['build_expansion_splits',
        '--source-splits',str(tmp_path/'source.json'),
        '--native-metadata',str(tmp_path/'native.json'),
        '--protected-protocol',str(tmp_path/'protocol.json'),
        '--data-root',str(tmp_path/'data'),'--out',str(out)])
    assert main()==0
    result=json.loads(out.read_text())
    selected=result['train_final']['train']
    assert {groups[n] for n in selected}=={'new_family_a','new_family_b'}
    assert len(selected)==2
    assert result['eligible_pages_before_family_balance']==3
    assert result['test_related_selected_source_groups']==1
