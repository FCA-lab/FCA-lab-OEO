"""Final-container semantics: indexing is removed, concept data are not copied."""
from types import SimpleNamespace
import sys
from oeo_lab.experiments.construction.evaluate import final_result_container


def test_flatten_preserves_original_concepts_and_excludes_empty_layers():
    bottom = [0, [0, 0]]
    top = [3, [1, 2]]
    builder = SimpleNamespace(ref={0: {0: bottom}, 1: {}, 2: {3: top}})
    result = final_result_container(builder, "3woc")
    assert set(result) == {0, 3}
    assert result[0] is bottom and result[3] is top
    before = sys.getsizeof(result)
    top[1].extend(range(1000))
    assert sys.getsizeof(result) == before  # referenced payload excluded


def test_qm_uses_existing_list():
    concepts = [object(), object()]
    assert final_result_container(SimpleNamespace(concepts=concepts), "qm") is concepts
