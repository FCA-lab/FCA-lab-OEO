"""Paper backgrounds: categorical scaling, or seeded legacy random generation."""
import csv
import random
from pathlib import Path

def random_context(objects, attributes, density, seed):
    if objects < 1 or attributes < 1 or not 0 <= density <= 1:
        raise ValueError("Invalid context dimensions/density")
    rng = random.Random(seed)
    matrix = [[0] * attributes for _ in range(objects)]
    remaining = round(objects * attributes * density)
    while remaining:
        i, j = rng.randrange(objects), rng.randrange(attributes)
        if matrix[i][j] == 0:
            matrix[i][j] = 1
            remaining -= 1
    return matrix

def load_context(spec):
    if spec["kind"] == "synthetic":
        return random_context(spec["objects"], spec["attributes"], spec["density"], spec["seed"])
    if spec["kind"] == "matrix":
        return spec["matrix"]
    lines = [line for line in Path(spec["path"]).read_text().splitlines() if line.strip()]
    if spec["format"] == "whitespace":
        rows = [line.split()[1:] for line in lines]  # Lenses: remove record ID only.
    else:
        rows = list(csv.reader(lines))
        if spec.get("header"):
            rows = rows[1:]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Malformed input rows")
    # Construction reproduction includes the original class column as an attribute.
    columns = [(j, value) for j in range(len(rows[0])) for value in sorted({row[j] for row in rows})]
    matrix = [[int(row[j] == value) for j, value in columns] for row in rows]
    if [len(matrix), len(columns)] != spec["expected_shape"]:
        raise ValueError("Encoded shape differs from paper context")
    return matrix
