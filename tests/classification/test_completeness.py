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
