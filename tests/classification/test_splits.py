import pytest
from oeo_lab.experiments.classification.splits import make_splits


def test_address_reproducible_and_full_coverage():
    y = ['a']*12 + ['b']*12
    first = make_splits(y,n_splits=3,seeds=[7,11])
    assert first == make_splits(y,n_splits=3,seeds=[7,11])
    for seed in [7,11]:
        tested = [i for r in first['splits'] if r['seed']==seed for i in r['test_indices']]
        assert sorted(tested) == list(range(24))


def test_entity_never_crosses_boundary():
    groups = [f'e{i}' for i in range(12) for _ in range(2)]
    y = ['a','b']*12
    result = make_splits(y,groups=groups,protocol='entity',n_splits=3,seeds=[7])
    for row in result['splits']:
        assert not ({groups[i] for i in row['train_indices']} & {groups[i] for i in row['test_indices']})
        assert row['entity_overlap']==0
        assert all(c['train'] and c['test'] for c in row['class_counts'])


@pytest.mark.parametrize('groups', [None, ['e1',None,'e2','e3'], ['e1','e1','e2','e2']])
def test_missing_or_insufficient_entities_rejected(groups):
    with pytest.raises(ValueError):
        make_splits(['a','a','b','b'],groups=groups,protocol='entity',n_splits=2)


def test_few_samples_rejected():
    with pytest.raises(ValueError):
        make_splits(['a','a','b'],n_splits=2)
