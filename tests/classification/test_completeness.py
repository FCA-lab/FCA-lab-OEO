import itertools
import numpy as np
from oeo_lab.experiments.classification.completeness import compare_context


def test_hand_calculated_rules_with_empty_support():
    result = compare_context([[1, 0], [0, 0]], ['a', 'b'])
    expected = {
        ('A1', 0, 0), ('A1', 1, 0), ('A2', 0, 1), ('A1', 1, 1),
        ('B1', 1, 0), ('B2', 0, 0), ('B1', 0, 1), ('B1', 1, 1),
    }
    assert set(result['direct_rules']) == expected
    assert set(result['full_rules']) == expected
    assert result['metrics']['comparison'] == 'pass'


def test_all_two_object_two_attribute_contexts_against_support_oracle():
    for cells in itertools.product([0, 1], repeat=4):
        X = np.array(cells).reshape(2, 2)
        for y in [['a', 'b'], ['a', 'a']]:
            result = compare_context(X, y)
            assert result['metrics']['comparison'] == 'pass', (X, y, result)


def test_object_permutation_does_not_change_rules():
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array(['a', 'b', 'a'])
    a = compare_context(X, y)
    b = compare_context(X[[2, 0, 1]], y[[2, 0, 1]])
    assert a['direct_rules'] == b['direct_rules']
    assert a['full_rules'] == b['full_rules']


def test_context_size_and_density():
    metrics = compare_context([[1,0],[0,0]], ['a','b'])['metrics']
    assert metrics['n_objects'] == 2
    assert metrics['n_attributes'] == 2
    assert metrics['density'] == 0.25


def test_raw_features_exclude_ids_and_decision(tmp_path):
    import hashlib
    from oeo_lab.experiments.classification.completeness import execute
    p = tmp_path/'tiny.csv'
    p.write_text('id1,red,yes,a\nid2,blue,no,b\n')
    spec = {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'format':'csv','decision_column':3,'id_columns':[0],'feature_kind':'categorical'}
    m = execute({'dataset':spec})['metrics']
    assert m['n_features_raw'] == 2
    assert m['n_objects'] == 2 and m['n_attributes'] == 4
    assert m['density'] == 0.5


def test_registry_is_only_dataset_selection_source(tmp_path, monkeypatch):
    import pytest
    from oeo_lab.experiments.classification import completeness as c
    p=tmp_path/'tiny.csv'
    p.write_text('1,a\n2,b\n')
    spec={'path':str(p),'format':'csv','decision_column':1,'id_columns':[],
          'feature_kind':'numerical'}
    monkeypatch.setattr(c.config,'DATASETS',{'new_numeric':spec})
    args,_=c.parse_args([])
    assert args.datasets==['new_numeric']
    tasks=c.build_tasks(args,{})
    assert tasks[0]['dataset']['max_bins']==3
    spec['preprocessing']={'max_bins':[5,10]}
    assert len(c.build_tasks(args,{}))==2
    args,_=c.parse_args(['--max-bins','20'])
    assert c.build_tasks(args,{})[0]['dataset']['max_bins']==20
    with pytest.raises(SystemExit):c.parse_args(['--datasets','wine'])
    monkeypatch.setattr(c.config,'DATASETS',{})
    with pytest.raises(SystemExit):c.parse_args([])
