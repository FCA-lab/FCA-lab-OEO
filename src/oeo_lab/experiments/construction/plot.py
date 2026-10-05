"""Plot construction summaries; adapted from legacy view/data_view.py bar3d."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_grids(summary, config):
    """Validate the long table, then pivot by explicit keys, never row order."""
    required = {'kind', 'objects', 'attributes', 'density', 'algorithm',
                'complete', 'completed_runs', 'expected_runs',
                'concept_count_mean', 'seconds_mean'}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f'Missing summary columns: {sorted(missing)}')
    frame = summary.loc[summary['kind'] == 'synthetic'].copy()
    keys = ['objects', 'attributes', 'density', 'algorithm']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate synthetic summary keys')
    spec = config['synthetic']
    objects = sorted(spec['objects'])
    densities = sorted(spec['densities'])
    grids = []
    for attributes in spec['attributes']:
        counts = []
        for algorithm in config['algorithms']:
            group = frame.loc[(frame['attributes'] == attributes) &
                              (frame['algorithm'] == algorithm)]
            expected = {(o, d) for o in objects for d in densities}
            if set(zip(group['objects'], group['density'])) != expected:
                raise ValueError(f'Incomplete or unexpected grid: {attributes}, {algorithm}')
            if not (group['complete'].eq(True).all() and
                    group['completed_runs'].eq(spec['repeats']).all() and
                    group['expected_runs'].eq(spec['repeats']).all()):
                raise ValueError(f'Incomplete repeats: {attributes}, {algorithm}')
            values = {}
            for metric in ['concept_count_mean', 'seconds_mean']:
                matrix = group.pivot(index='objects', columns='density', values=metric)
                z = matrix.reindex(index=objects, columns=densities).to_numpy(dtype=float)
                if not np.isfinite(z).all() or (z < 0).any():
                    raise ValueError(f'Invalid values: {metric}, {attributes}, {algorithm}')
                values[metric] = z
            counts.append(values['concept_count_mean'])
            grids.append(('time', attributes, algorithm, objects, densities,
                          values['seconds_mean']))
        if not all(np.array_equal(counts[0], z) for z in counts[1:]):
            raise ValueError(f'Concept counts disagree between algorithms: {attributes}')
        grids.append(('concepts', attributes, 'shared', objects, densities, counts[0]))
    return grids


def draw_3d_bar(objects, densities, values, title, zlabel, paths, dpi):
    # Preserve the legacy bar3d design, with numeric arrays instead of eval/CSV cells.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x, y = np.meshgrid(objects, densities, indexing='ij')
    dx = min(np.diff(objects)) * .5 if len(objects) > 1 else .5
    dy = min(np.diff(densities)) * .5 if len(densities) > 1 else .05
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x.ravel()-dx/2, y.ravel()-dy/2, np.zeros(values.size),
             dx, dy, values.ravel(), color='deepskyblue', zsort='average', alpha=.85)
    ax.set_xlabel('Number of objects', labelpad=10)
    ax.set_ylabel('Density', labelpad=10)
    ax.set_zlabel(zlabel, labelpad=12)
    ax.set_xticks(objects)
    ax.set_yticks(densities)
    ax.set_zlim(bottom=0)
    ax.set_title(title)
    ax.view_init(elev=25, azim=-55)
    fig.subplots_adjust(left=.04, right=.85, bottom=.08, top=.90)
    try:
        for path in paths:
            fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.35)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(__file__).resolve().parents[4]
    output = root / config['output']['root']
    workbook = output / config['output']['files']['results']['synthetic']
    settings = config['plot']
    grids = prepare_grids(pd.read_excel(workbook, sheet_name='summary'), config)
    # Read-only postprocessing: no experiment execution or checkpoint changes.
    destination = output / settings['directory']
    destination.mkdir(parents=True, exist_ok=True)
    labels = {'3woc': '3WOC', 'qm': 'QM'}
    for metric, attributes, algorithm, objects, densities, values in grids:
        name = settings['files'][metric].format(attributes=attributes, algorithm=algorithm)
        paths = [destination / (name + '.' + ext) for ext in settings['formats']]
        title = f'{attributes} attributes'
        zlabel = 'Mean number of concepts'
        if metric == 'time':
            title = f'{labels.get(algorithm, algorithm)} — {title}'
            zlabel = 'Mean running time (s)'
        draw_3d_bar(objects, densities, values, title, zlabel, paths, settings['dpi'])
        for path in paths:
            print(path)


if __name__ == '__main__':
    main()
