"""Frozen legacy outputs: these test migration parity, not mathematical validity."""
import json
from pathlib import Path
import pytest
from fca_core.context import formal_context
from fca_core.data_source import data_set
from oeo_lab.algorithms.construction.three_woc import NewThreeWayForObj
from oeo_lab.algorithms.construction.juxtaposition import ThreeWayOC

CASES=json.loads((Path(__file__).parents[1]/'fixtures/legacy_construction.json').read_text())
@pytest.mark.parametrize('case', CASES)
def test_preserved_construction_output(case):
    cls=NewThreeWayForObj if case['algorithm']=='3woc' else ThreeWayOC
    obj=cls(formal_context(data_set(case['input']),1),1)
    obj.start()
    if hasattr(obj,'ref'):
        actual=sorted([int(r[0]),[int(r[1][0]),int(r[1][1])]] for b in obj.ref.values() for r in b.values())
    else:
        actual=sorted([sorted(c.extent),sorted(c.intent),sorted(c.modus)] for c in obj.concepts)
    assert actual==case['expected']
