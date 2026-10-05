import math
import pytest
from sklearn.metrics import accuracy_score, f1_score
from oeo_lab.experiments.classification.evaluate import evaluate_predictions


def test_all_correct():
    r = evaluate_predictions(['a','b'], ['a','b'], ['a','b'])
    assert (r['coverage'], r['classified_accuracy'], r['overall_correct_rate'], r['macro_f1']) == (1,1,1,1)


def test_partial_defer_hand_calculated():
    r = evaluate_predictions(['a','a','b','b'], ['a',None,'a','b'], ['a','b'])
    assert (r['n_test'], r['n_classified'], r['n_correct'], r['n_deferred']) == (4,3,2,1)
    assert r['coverage'] == .75
    assert r['classified_accuracy'] == pytest.approx(2/3)
    assert r['overall_correct_rate'] == .5
    assert r['macro_f1'] == pytest.approx((.5 + 2/3)/2)
    assert r['per_class'][0]['fn'] == 1
    assert r['per_class'][1]['fp'] == 0


def test_all_defer():
    r = evaluate_predictions([0,1], [None,None], [0,1])
    assert math.isnan(r['classified_accuracy'])
    assert r['coverage'] == r['overall_correct_rate'] == r['macro_f1'] == 0


def test_fixed_class_set_and_missed_class():
    r = evaluate_predictions([0,1], [0,0], [0,1,2])
    assert r['macro_f1'] == pytest.approx(2/9)
    assert r['per_class'][2]['support'] == 0


def test_baseline_matches_sklearn():
    y = [0,0,1,1,2]
    p = [0,1,1,2,2]
    r = evaluate_predictions(y,p,[0,1,2])
    assert r['coverage'] == 1
    assert r['classified_accuracy'] == r['overall_correct_rate'] == accuracy_score(y,p)
    assert r['macro_f1'] == pytest.approx(f1_score(y,p,labels=[0,1,2],average='macro'))


@pytest.mark.parametrize('y,p,labels', [([],[],[0]), ([0],[0,0],[0]),
    ([0],[2],[0,1]), ([None],[0],[0]), ([0],[float('nan')],[0]), ([0],[0],[0,0])])
def test_invalid_inputs(y,p,labels):
    with pytest.raises(ValueError):
        evaluate_predictions(y,p,labels)
