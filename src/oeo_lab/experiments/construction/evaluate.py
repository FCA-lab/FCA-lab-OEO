"""Adapters around unmodified construction algorithms."""
import contextlib
import os
import sys
import time
from .datasets import load_context
from ..common.checkpoints import digest

def final_result_container(builder, algorithm):
    """Flatten only the 3WOC index; retain references to original concepts."""
    if algorithm == "3woc":
        return {extent: concept for bucket in builder.ref.values()
                for extent, concept in bucket.items()}
    return builder.concepts


def mask(values):
    return sum(1 << int(v) for v in values)

def execute(task):
    from threadpoolctl import threadpool_limits
    from fca_core.context import formal_context
    from fca_core.data_source import data_set
    from oeo_lab.algorithms.construction.three_woc import NewThreeWayForObj
    from oeo_lab.algorithms.construction.juxtaposition import ThreeWayOC
    matrix = load_context(task["background"])
    context = formal_context(data_set(matrix), 1)
    with threadpool_limits(limits=1):
        builder = (NewThreeWayForObj if task["algorithm"] == "3woc" else ThreeWayOC)(context, 1)
        # Retain algorithm code; suppress console I/O. print formatting overhead remains.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            start = time.perf_counter()
            builder.start()
            seconds = time.perf_counter() - start
    # Result-container overhead only: no recursion or intermediate structures.
    # Aggregation and measurement are both outside start() timing.
    container = final_result_container(builder, task["algorithm"])
    container_bytes = sys.getsizeof(container)
    if task["algorithm"] == "3woc":
        concepts = sorted((int(c[0]), int(c[1][0]), int(c[1][1])) for bucket in builder.ref.values() for c in bucket.values())
        container_name = "flat dict (shallow)"
    else:
        concepts = sorted((mask(c.extent), mask(c.intent), mask(c.modus)) for c in builder.concepts)
        container_name = "concepts list (shallow)"
    full = (1 << len(matrix)) - 1
    supports = [mask(i for i, row in enumerate(matrix) if row[j]) for j in range(len(matrix[0]))]
    invalid = 0
    for extent, positive, negative in concepts:
        expected_pos = mask(j for j,s in enumerate(supports) if s & ~extent == 0)
        expected_neg = mask(j for j,s in enumerate(supports) if (full ^ s) & ~extent == 0)
        recovered = 0
        for j,s in enumerate(supports):
            if positive >> j & 1: recovered |= s
            if negative >> j & 1: recovered |= full ^ s
        invalid += (positive != expected_pos or negative != expected_neg or recovered != extent)
    return {"background_sha256": digest(matrix), "objects": len(matrix), "attributes": len(matrix[0]),
            "actual_density": sum(map(sum, matrix)) / (len(matrix)*len(matrix[0])),
            "seconds": seconds, "concept_count": len(concepts), "invalid_concepts": invalid,
            "duplicate_concepts": len(concepts)-len(set(concepts)),
            "result_container_bytes": container_bytes, "result_container": container_name,
            "concepts": concepts}
