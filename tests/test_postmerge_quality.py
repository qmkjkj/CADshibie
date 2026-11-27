from types import SimpleNamespace

import numpy as np

from cadvec.schema import assemble_output
from tools.train_postmerge_quality import (
    FEATURES, calibrate, feature_vector, make_merge, quality_probability, score_filtered, split_families,
)


def output(length=10, class_id=2, instance=0):
    return assemble_output([SimpleNamespace(command=0, points=[[1, 1], [2, 1], [3, 1], [4, 1]],
        length=length, semantic_id=class_id, instance_id=instance, rgb=[0, 0, 0], layer_id=0, width=1)], width=100, height=100)


def model(probability=.75):
    return {'features': list(FEATURES), 'transform': 'log1p_nonnegative_except_aspect',
        'mean': [0.]*9, 'scale': [1.]*9, 'coef': [0.]*9,
        'intercept': float(np.log(probability/(1-probability))), 'threshold': .5}


def test_portable_quality_transform_and_probability():
    features = dict(zip(FEATURES, [3., 10., 4., 6., 5., 2., 10., .4, 1.5]))
    vector = feature_vector(features)
    assert vector[7] == .4 and vector[1] == np.log1p(10.)
    assert abs(quality_probability(features, model())-.75) < 1e-12


def test_calibration_reassigns_full_page_after_filtering_and_breaks_ties_strictly():
    # Removing the first candidate allows a different one-to-one match; max-IoU
    # independent candidate labels would double-count the first truth.
    matrix = np.array([[.9, .8], [.7, 0.]])
    score = score_filtered(matrix, [True, True])
    assert score['TP'] == 2 and score['sum_iou'] == 1.5
    score = score_filtered(matrix, [False, True])
    assert score['TP'] == 1 and score['sum_iou'] == .8
    best, scans = calibrate({'page': {'gt_count': 2, 'pred_count': 2, 'ious': matrix.tolist()}}, {'page': [.9, .9]})
    assert len(scans) == 19 and best['threshold'] == .9


def test_wrapper_baseline_reproduces_old_gate_and_learned_applies_only_to_native():
    empty = assemble_output([], width=100, height=100)
    options = {'gat_classes': set(), 'segdet_classes': set(), 'gate_classes': set(),
        'segdet_union_classes': {2}, 'suppress': set(), 'quality_rules': {2: ('total', '>=', 22.)},
        'source_identity': 'legacy'}
    native = output()
    assert make_merge()(empty, None, native, **options) == empty
    kept = make_merge(model())(empty, None, native, **options)
    assert kept == native
    # An unrelated primary instance still uses the original length rule.
    primary = output(class_id=2, instance=4)
    assert make_merge(model())(primary, None, empty, **options) == empty


def test_fit_and_calibration_families_are_disjoint():
    pages = ['a', 'b', 'c', 'd', 'e', 'f']; groups = {s: s.upper() for s in pages}
    groups['b'] = groups['a']
    fit, calib = split_families(pages, groups)
    assert set(fit) | set(calib) == set(pages)
    assert not {groups[s] for s in fit} & {groups[s] for s in calib}
