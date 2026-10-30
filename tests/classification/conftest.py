"""Synthetic input files for data-independent workflow tests."""
from itertools import product
import pytest


@pytest.fixture
def synthetic_uci_data(tmp_path, monkeypatch):
    """Keep public dataset schemas, without copying research datasets."""
    from oeo_lab.experiments.classification import uci_config, training_size_config
    root = tmp_path / "synthetic_data"
    rows = [[c * 10 + i / 10 + j for j in range(4)] + [c]
            for c in range(3) for i in range(20)]
    def write(relative, values, sep=","):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sep.join(map(str, row)) for row in values) + "\n")
    write("raw/uci/iris/iris.data", rows)
    write("raw/uci/wine/wine.data", [[r[-1]] + r[:-1] for r in rows])
    write("raw/uci/wdbc/wdbc.data", [[i, r[-1]] + r[:-1] for i, r in enumerate(rows)])
    write("raw/uci/balloons/adult+stretch.data",
          [[a, b, c, d, a] for a, b, c, d in product(range(2), repeat=4)])
    # Complete categorical grid: 3+2+2+2 encoded attributes; a binary attribute supplies negative-literal evidence.
    write("raw/uci/lenses/lenses.data",
          [[i, a, b, c, d, 0 if b == 0 else (1 if a == 0 else 2)] for i, (a, b, c, d) in
           enumerate(product(range(3), range(2), range(2), range(2)))], sep=" ")
    for config in (uci_config, training_size_config):
        monkeypatch.setattr(config, "DATA_ROOT", root)
    return len(rows)
