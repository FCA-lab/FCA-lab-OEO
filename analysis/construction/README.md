# Construction analysis

The maintained 3D plotter is `src/oeo_lab/experiments/construction/plot.py`. Run `PYTHONPATH=src python -m oeo_lab.experiments.construction.plot`; settings live alongside the experiment in `config.json`. It consumes the synthetic workbook summary and saves figures under the configured output directory.

`plot_legacy.py` is retained as a historical 2D CSV plotter; it does not read current workbooks.
