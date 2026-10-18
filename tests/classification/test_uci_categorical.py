"""类别预处理复用、训练折隔离及 UCI 三类实验共用流程。"""
import numpy as np
import pandas as pd
from oeo_lab.preprocessing.categorical import CategoricalContextEncoder, encode_categorical
from oeo_lab.experiments.classification import uci_experiment as u
from oeo_lab.experiments.classification.evaluate import evaluate_fold
from oeo_lab.experiments.classification.completeness import encode_categorical as complete_encode


def test_encoding_shared_and_unknown_test_category():
    train = pd.DataFrame({0: ['a','b','a'], 2: ['x','x','y']})
    test = pd.DataFrame({0: ['unseen'], 2: ['y']})
    encoder = CategoricalContextEncoder()
    actual = encoder.fit_transform(train)
    expected, metadata = complete_encode(train)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(expected, [[1,0,1,0],[0,1,1,0],[1,0,0,1]])
    np.testing.assert_array_equal(encoder.transform(test), [[0,0,0,1]])
    assert encoder.categories_ == metadata['columns']
    assert len(encoder.columns_) == 4  # 测试集未见类别不新增训练属性。


def test_categorical_tasks_no_bins_and_dataset_folds():
    args = u.parse_args(['--datasets','balloons','lenses','--models','rc-oeo',
                        'decision_tree','--seeds','11','--max-bins','20','50'])
    tasks = u.build_tasks(args, {'test': True})
    assert len(tasks) == (5+4)*2
    assert all(t['max_bins'] is None for t in tasks)
    for name, folds in [('balloons',5),('lenses',4)]:
        selected = [t for t in tasks if t['name'] == name]
        assert len({t['split']['fold'] for t in selected}) == folds
        for fold in range(1,folds+1):
            rows = [t for t in selected if t['split']['fold'] == fold]
            assert rows[0]['split'] == rows[1]['split']
    task = next(t for t in tasks if t['name']=='lenses' and t['model_name']=='rc-oeo')
    result = evaluate_fold(task)
    assert result['attributes'] == 9  # 排除 ID 和类别列。
    assert set(r['study'] for r in result['results']) == {'comparison','sensitivity','ablation'}
    assert result['encoding']['bins'] == {}
    assert any(r['type']=='A2' for r in result['rules'])
    assert any(r['type']=='B2' for r in result['rules'])
    baseline = next(t for t in tasks if t['name']=='lenses' and t['model_name']=='decision_tree')
    assert evaluate_fold(baseline)['attributes'] == result['attributes']
