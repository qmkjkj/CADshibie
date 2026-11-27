from types import SimpleNamespace

import pytest

from cadvec.schema import assemble_output
from tools.diagnose_fusion_losses import diagnose_page, last_loss_stage, STAGES


def output(length=10):
    return assemble_output([SimpleNamespace(command=0, points=[[1, 1], [2, 1], [3, 1], [4, 1]],
        length=length, semantic_id=2, instance_id=0, rgb=[0, 0, 0], layer_id=0, width=1)], width=100, height=100)


def test_fresh_trace_geometry_replaces_old_snapshot_and_attributes_quality_deletion():
    empty = assemble_output([], width=100, height=100)
    current = output()
    snapshot = {'primary': empty, 'gat': None, 'segdet': empty, 'merged': output(999), 'final': output(999),
        'support': {'size': [1000, 1000], 'detections': []}, 'stuff_items': [],
        'merge_options': {'gat_classes': [], 'segdet_classes': [], 'gate_classes': [],
            'segdet_union_classes': [2], 'suppress': [], 'quality_rules': {'2': ['total', '>=', 22.]},
            'raster_mode': 'legacy400'},
        'assignment': {'anchor_classes': [], 'atom_classes': [], 'atom_affinity': None}}
    trace = {'verified_current_baseline': True, 'source_identity': 'legacy', 'segdet': current, 'final': empty}
    result = diagnose_page(snapshot, trace, empty, current)
    assert result['segdet_to_final'] == {'lost': 1, 'gained': 0}
    case = result['lost_cases'][0]
    assert case['cause'] == 'quality_rule_deletion'
    assert case['quality_rule']['value'] == 10 and not case['quality_rule']['passes']
    assert case['stages']['merge_without_quality']['matched']
    with pytest.raises(ValueError, match='differs'):
        diagnose_page(snapshot, dict(trace, final=current), current, current)


def test_attribution_uses_last_loss_after_an_intermediate_recovery():
    values = dict(zip(STAGES, [True, False, True, False, False, False]))
    assert last_loss_stage(values) == 'anchor'
